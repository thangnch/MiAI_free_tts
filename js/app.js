import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
import initG2p, * as g2pModule from "../g2p/sea_g2p.js";
import { VieneuNanoWeb, loadVoices, encodeWav, NANO_HF_BASE } from "./tts.js";
import { fetchCached } from "./nano.js";
import { joinChunks } from "./audio.js";
import { GAP_SILENCE } from "./chunker.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const MODEL_BASE = params.get("models") || NANO_HF_BASE;   // ?models=./models để tự host model
const FORCE_EP = params.get("ep");                         // ?ep=wasm để ép chạy CPU

let tts = null, abortCtl = null, lastBlobUrl = null;

const fmtMB = (b) => (b / 1048576).toFixed(1) + " MB";
function setStatus(msg, kind = "") { const s = $("status"); s.textContent = msg; s.dataset.kind = kind; }
function setProgress(frac) { $("bar").style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`; }

async function gunzip(buf) {
  const ds = new DecompressionStream("gzip");
  return new Uint8Array(await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer());
}

async function pickEp() {
  if (FORCE_EP) return FORCE_EP;
  if (!("gpu" in navigator)) return "wasm";
  try { return (await navigator.gpu.requestAdapter()) ? "webgpu" : "wasm"; } catch { return "wasm"; }
}

async function boot() {
  const ep = await pickEp();
  $("ep").textContent = ep === "webgpu" ? "WebGPU" : "CPU (WASM)";
  $("ep").dataset.kind = ep;
  if (ep === "wasm") {
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
  }
  const t0 = performance.now();
  try {
    setStatus("Đang tải bộ phiên âm tiếng Việt…");
    const [, dictGz, vj, vf] = await Promise.all([
      initG2p({ module_or_path: new URL("../g2p/sea_g2p_bg.wasm", import.meta.url) }),
      fetchCached(new URL("../g2p/sea_g2p_vi.bin.gz", import.meta.url).href, {
        onProgress: (g, t) => { setStatus(`Đang tải từ điển phiên âm… ${fmtMB(g)}${t ? " / " + fmtMB(t) : ""}`); setProgress(t ? (g / t) * 0.25 : 0); },
      }),
      fetch(new URL("../voices/voices_v3_nano.json", import.meta.url)).then((r) => r.arrayBuffer()),
      fetch(new URL("../voices/voices_v3_nano.f32", import.meta.url)).then((r) => r.arrayBuffer()),
    ]);
    const dictBytes = await gunzip(dictGz);
    const voices = await loadVoices(vj, vf);

    const create = (epName) => VieneuNanoWeb.create({
      ort, g2pModule, dictBytes, voices, modelBase: MODEL_BASE, ep: epName,
      onProgress: (g, t) => { setStatus(`Đang tải model Nano… ${fmtMB(g)} / ${fmtMB(t)} (chỉ tải lần đầu)`); setProgress(0.25 + (t ? (g / t) * 0.7 : 0)); },
    });
    try { tts = await create(ep); }
    catch (e) {
      if (ep !== "webgpu") throw e;
      console.warn("WebGPU lỗi, chuyển sang WASM:", e);
      $("ep").textContent = "CPU (WASM)"; $("ep").dataset.kind = "wasm";
      tts = await create("wasm");
    }

    const sel = $("voice");
    for (const [label, name] of tts.listVoices()) {
      const o = document.createElement("option"); o.value = name; o.textContent = label; sel.appendChild(o);
    }
    sel.value = params.get("voice") || "Mạnh Dũng";
    if (!sel.value) sel.value = voices.default_voice;

    setProgress(1);
    setStatus(`Sẵn sàng sau ${((performance.now() - t0) / 1000).toFixed(1)} s. Mọi xử lý chạy trên máy bạn.`, "ok");
    $("go").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("Không khởi tạo được: " + (e.message || e), "err");
  }
}

// Chunk-level playback through Web Audio while the rest is still being generated.
class StreamPlayer {
  constructor(sr) { this.ctx = new AudioContext({ sampleRate: sr }); this.t = 0; this.sr = sr; this.nodes = []; }
  push(wav, padBefore) {
    const start = Math.max(this.t, this.ctx.currentTime + 0.05) + padBefore / this.sr;
    const b = this.ctx.createBuffer(1, wav.length, this.sr); b.copyToChannel(wav, 0);
    const n = this.ctx.createBufferSource(); n.buffer = b; n.connect(this.ctx.destination); n.start(start);
    this.nodes.push(n); this.t = start + wav.length / this.sr;
  }
  stop() { for (const n of this.nodes) { try { n.stop(); } catch {} } this.ctx.close(); }
}
let player = null;

async function generate() {
  const text = $("text").value.trim();
  if (!text || !tts) return;
  const fast = $("fast").checked;
  const opts = {
    voice: $("voice").value,
    steps: fast ? 8 : 16, sway: fast ? -1 : 0,
    cfg: Number($("cfg").value),
    speed: Number($("speed").value),
    seed: $("seed").value === "" ? null : Number($("seed").value),
  };
  abortCtl = new AbortController();
  $("go").disabled = true; $("stop").disabled = false; $("result").hidden = true;
  player?.stop(); player = $("stream").checked ? new StreamPlayer(tts.sampleRate) : null;

  const t0 = performance.now();
  const wavs = [];
  try {
    const { gaps } = tts.prepare(text);
    for await (const r of tts.inferStream(text, {
      ...opts, signal: abortCtl.signal,
      onStep: (ci, cn, s, n) => { setProgress((ci + s / n) / cn); setStatus(`Đang tổng hợp đoạn ${ci + 1}/${cn} — bước ${s}/${n}`); },
    })) {
      wavs.push(r.wav);
      if (player && r.wav.length) player.push(r.wav, r.padBefore);
    }
    const wav = joinChunks(wavs, tts.sampleRate, gaps.map((g) => GAP_SILENCE[g] ?? GAP_SILENCE.sentence));
    const secs = (performance.now() - t0) / 1000, dur = wav.length / tts.sampleRate;
    const blob = encodeWav(wav, tts.sampleRate);
    if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
    lastBlobUrl = URL.createObjectURL(blob);
    $("audio").src = lastBlobUrl;
    $("download").href = lastBlobUrl;
    $("download").download = `vieneu_nano_${Date.now()}.wav`;
    $("stats").textContent = `${dur.toFixed(1)} s audio trong ${secs.toFixed(2)} s · RTF ${(secs / dur).toFixed(2)} · ${wavs.length} đoạn · ${$("ep").textContent}`;
    $("result").hidden = false;
    if (!player) $("audio").play().catch(() => {});
    setProgress(1); setStatus("Xong.", "ok");
  } catch (e) {
    if (e.name === "AbortError") setStatus("Đã dừng.");
    else { console.error(e); setStatus("Lỗi: " + (e.message || e), "err"); }
  } finally {
    $("go").disabled = false; $("stop").disabled = true;
  }
}

$("go").addEventListener("click", generate);
$("stop").addEventListener("click", () => { abortCtl?.abort(); player?.stop(); player = null; });
$("speed").addEventListener("input", (e) => ($("speedv").textContent = Number(e.target.value).toFixed(2) + "×"));
$("cfg").addEventListener("input", (e) => ($("cfgv").textContent = Number(e.target.value).toFixed(1)));
$("text").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate(); });
document.querySelectorAll("[data-cue]").forEach((b) => b.addEventListener("click", () => {
  const ta = $("text"), p = ta.selectionStart ?? ta.value.length;
  ta.setRangeText(b.dataset.cue, p, ta.selectionEnd ?? p, "end"); ta.focus();
}));
boot();

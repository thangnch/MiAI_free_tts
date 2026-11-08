// VieNeu-TTS v3 Nano engine for onnxruntime-web — port of vieneu/v3nano.py::OnnxV3NanoEngine.
import { loadNpz } from "./npz.js";
import { trimAndFade } from "./audio.js";

export const NANO_HF_BASE = "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Nano/resolve/main";
const GRAPHS = ["text_encoder", "duration_predictor", "vector_estimator", "codec_decoder"];
const MAX_CHUNK_SECONDS = 15.0;
const MIN_FRAMES = 2;
const LATENT_CH = 144;
const f = Math.fround;
/** Python round(): half to even */
const pyRound = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };

/** Seeded N(0,1) generator (mulberry32 + Box–Muller). */
export function makeRng(seed) {
  let a = (seed ?? Math.floor(Math.random() * 2 ** 32)) >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  return {
    normal() {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u1 = u(); while (u1 <= 0) u1 = u();
      const u2 = u();
      const r = Math.sqrt(-2 * Math.log(u1));
      spare = r * Math.sin(2 * Math.PI * u2);
      return r * Math.cos(2 * Math.PI * u2);
    },
  };
}

/** fetch with progress + Cache API (so each user downloads the models once). */
export async function fetchCached(url, { cacheName = "vieneu-nano-v1", onProgress } = {}) {
  let cache = null;
  try { cache = await caches.open(cacheName); } catch { /* no Cache API (http/file) */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { const b = await hit.arrayBuffer(); onProgress?.(b.byteLength, b.byteLength); return b; }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} khi tải ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  let buf;
  if (res.body && onProgress) {
    const reader = res.body.getReader();
    const parts = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length; onProgress(got, total);
    }
    const u8 = new Uint8Array(got); let o = 0;
    for (const p of parts) { u8.set(p, o); o += p.length; }
    buf = u8.buffer;
  } else buf = await res.arrayBuffer();
  if (cache) { try { await cache.put(url, new Response(buf.slice(0))); } catch { /* quota */ } }
  return buf;
}

export class NanoEngine {
  static SAMPLE_RATE = 24000;

  /**
   * @param ort       onnxruntime-web module
   * @param opts.base URL folder with the 4 .onnx graphs + config.json + constants.npz
   * @param opts.ep   "webgpu" | "wasm"
   */
  static async create(ort, { base = NANO_HF_BASE, ep = "webgpu", onProgress, fetcher = fetchCached } = {}) {
    const e = new NanoEngine();
    e.ort = ort; e.ep = ep;
    const files = [...GRAPHS.map((g) => g + ".onnx"), "config.json", "constants.npz"];
    const done = {}, totals = {};
    const report = (f) => (got, tot) => { done[f] = got; totals[f] = tot || got;
      onProgress?.(Object.values(done).reduce((a, b) => a + b, 0), Object.values(totals).reduce((a, b) => a + b, 0), f); };
    const bufs = await Promise.all(files.map((f) => fetcher(`${base}/${f}`, { onProgress: report(f) })));
    const byName = Object.fromEntries(files.map((f, i) => [f, bufs[i]]));

    e.cfg = JSON.parse(new TextDecoder().decode(byName["config.json"]));
    const c = await loadNpz(byName["constants.npz"]);
    e.nullSpk = c.null_spk.data;                       // [192]
    e.nullStyle = c.null_style;                        // {data, shape:[S,C]}
    e.vocab = e.cfg.vocab;
    e.bos = e.cfg.bos_id | 0; e.eos = e.cfg.eos_id | 0; e.pad = e.cfg.pad_id | 0;
    e.fps = Number(e.cfg.flow_fps ?? 24000 / 256 / 6);
    e.emotionMap = e.cfg.emotion_tags || {};
    e.oov = new Set();

    const so = { executionProviders: ep === "webgpu" ? ["webgpu"] : ["wasm"], graphOptimizationLevel: "all" };
    e.sess = {};
    for (const g of GRAPHS) e.sess[g] = await ort.InferenceSession.create(new Uint8Array(byName[g + ".onnx"]), so);

    const [S, C] = e.nullStyle.shape.slice(-2);
    e.styleShape = [S, C];
    e.nullStyleT = new ort.Tensor("float32", e.nullStyle.data, [1, S, C]);
    e.nullSpkT = new ort.Tensor("float32", e.nullSpk, [1, e.nullSpk.length]);
    const nullIds = e.idsTensor([e.bos, e.eos]);
    e.nullCtx = (await e.sess.text_encoder.run({ ids: nullIds, style: e.nullStyleT }))[e.sess.text_encoder.outputNames[0]];
    e.nullMask = e.maskTensor([e.bos, e.eos]);
    return e;
  }

  idsTensor(ids) { return new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]); }
  maskTensor(ids) { return new this.ort.Tensor("bool", Uint8Array.from(ids, (i) => (i !== this.pad ? 1 : 0)), [1, ids.length]); }

  /** phone string → ids (unknown chars dropped, like the Python engine). */
  encodePhones(phones) {
    let s = phones;
    for (const [tag, ch] of Object.entries(this.emotionMap)) s = s.split(tag).join(ch);
    const ids = [this.bos];
    for (const ch of s) {
      const i = this.vocab[ch];
      if (i === undefined) {
        if (!this.oov.has(ch)) { this.oov.add(ch); console.warn("v3 Nano: bỏ qua ký tự ngoài vocab", ch); }
        continue;
      }
      ids.push(i);
    }
    ids.push(this.eos);
    return ids;
  }

  async run(name, feeds) {
    const s = this.sess[name];
    const out = await s.run(feeds);
    return out[s.outputNames[0]];
  }

  /**
   * One chunk of phonemes → Float32Array @ 24 kHz.
   * @param spk   Float32Array(192)   @param style Float32Array(S*C)
   */
  async infer(phonemes, spk, style, { steps = 16, cfg = 3.0, sway = 0.0, speed = 1.0, rng, noise, onStep } = {}) {
    const ort = this.ort;
    const [S, C] = this.styleShape;
    const spkT = new ort.Tensor("float32", spk, [1, spk.length]);
    const styleT = new ort.Tensor("float32", style, [1, S, C]);
    const ids = this.encodePhones(phonemes);
    const idsT = this.idsTensor(ids), maskT = this.maskTensor(ids);

    const ctx = await this.run("text_encoder", { ids: idsT, style: styleT });
    const logS = (await this.run("duration_predictor", { ctx, ctx_mask: maskT, spk: spkT })).data[0];
    const secs = Math.min(Math.exp(Number(logS)) / Math.max(speed, 1e-3), MAX_CHUNK_SECONDS);
    const T = Math.max(MIN_FRAMES, pyRound(secs * this.fps));

    const N = LATENT_CH * T;
    let x;
    if (noise) x = Float32Array.from(noise(N));
    else { x = new Float32Array(N); const r = rng || makeRng(); for (let i = 0; i < N; i++) x[i] = r.normal(); }

    const n = Math.max(1, steps | 0);
    const tg = new Float64Array(n + 1);
    for (let i = 0; i <= n; i++) {
      const u = i === n ? 1 : i * (1 / n);            // == np.linspace(0, 1, n + 1)
      tg[i] = u + sway * (Math.cos((Math.PI / 2) * u) - 1 + u);
    }
    for (let i = 0; i < n; i++) {
      const tT = new ort.Tensor("float32", Float32Array.of(tg[i]), [1]);
      const xT = new ort.Tensor("float32", x, [1, LATENT_CH, T]);
      const v = (await this.run("vector_estimator", { x: xT, t: tT, ctx, ctx_mask: maskT, spk: spkT, style: styleT })).data;
      let vel = v;
      if (cfg > 0) {
        const vu = (await this.run("vector_estimator", { x: xT, t: tT, ctx: this.nullCtx, ctx_mask: this.nullMask,
          spk: this.nullSpkT, style: this.nullStyleT })).data;
        vel = new Float32Array(N);
        const c32 = f(cfg);
        for (let k = 0; k < N; k++) vel[k] = vu[k] + f(c32 * f(v[k] - vu[k]));   // float32 at each op, like NumPy
      }
      const dt = Math.fround(tg[i + 1] - tg[i]);
      const nx = new Float32Array(N);
      for (let k = 0; k < N; k++) nx[k] = x[k] + f(dt * vel[k]);
      x = nx;
      onStep?.(i + 1, n);
    }
    const wav = (await this.run("codec_decoder", { x: new ort.Tensor("float32", x, [1, LATENT_CH, T]) })).data;
    const out = new Float32Array(wav.length);
    for (let k = 0; k < wav.length; k++) out[k] = Math.max(-1, Math.min(1, wav[k]));
    return trimAndFade(out, NanoEngine.SAMPLE_RATE);
  }
}

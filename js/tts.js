// High-level API mirroring `Vieneu(mode="v3nano").infer(...)`, fully client-side.
import { normalizeToChunks, phonemizeWithEmotions, GAP_SILENCE } from "./chunker.js";
import { NanoEngine, makeRng, fetchCached, NANO_HF_BASE } from "./nano.js";
import { joinChunks, pausePadSamples, encodeWav } from "./audio.js";

export { encodeWav, NANO_HF_BASE };

export class VieneuNanoWeb {
  /**
   * @param o.ort        onnxruntime-web module
   * @param o.g2pModule  wasm-bindgen module of sea-g2p (already initialised)
   * @param o.dictBytes  Uint8Array of sea_g2p_vi.bin
   * @param o.voices     { default_voice, list:[{name, description, spk:Float32Array, style:Float32Array}] }
   */
  static async create({ ort, g2pModule, dictBytes, voices, modelBase = NANO_HF_BASE, ep = "webgpu", onProgress, fetcher }) {
    const t = new VieneuNanoWeb();
    t.g2p = new g2pModule.SeaG2p(dictBytes);
    t.puncNorm = g2pModule.puncNorm;
    t.voices = voices;
    t.engine = await NanoEngine.create(ort, { base: modelBase, ep, onProgress, fetcher: fetcher || fetchCached });
    t.sampleRate = NanoEngine.SAMPLE_RATE;
    return t;
  }

  listVoices() { return this.voices.list.map((v) => [v.description ? `${v.name} — ${v.description}` : v.name, v.name]); }
  voice(name) {
    const v = this.voices.list.find((x) => x.name === (name || this.voices.default_voice));
    if (!v) throw new Error(`Không có giọng '${name}'`);
    return v;
  }

  /** text → [{text, phonemes}], gaps */
  prepare(text, maxChars = 140) {
    const [chunks, gaps] = normalizeToChunks(text, this.g2p, this.puncNorm, maxChars);
    return { chunks: chunks.map((c) => ({ text: c, phonemes: phonemizeWithEmotions(c, this.g2p, this.puncNorm) })), gaps };
  }

  /**
   * Chunk-level streaming: yields {index, total, wav, padBefore} as each chunk finishes.
   * Same options as Python: voice, steps(16), cfg(3.0), sway(0), speed(1), seed, maxChars(140).
   */
  async *inferStream(text, { voice, steps = 16, cfg = 3.0, sway = 0, speed = 1.0, seed = null, maxChars = 140, onStep, noiseFor, signal } = {}) {
    const v = this.voice(voice);
    const { chunks, gaps } = this.prepare(text, maxChars);
    let prev = null;
    for (let i = 0; i < chunks.length; i++) {
      if (signal?.aborted) throw new DOMException("Đã huỷ", "AbortError");
      const rng = makeRng(seed == null ? undefined : seed + i);
      const wav = await this.engine.infer(chunks[i].phonemes, v.spk, v.style, {
        steps, cfg, sway, speed, rng, noise: noiseFor?.(i), onStep: (s, n) => onStep?.(i, chunks.length, s, n),
      });
      const pause = i > 0 ? GAP_SILENCE[gaps[i - 1]] ?? GAP_SILENCE.sentence : 0;
      const padBefore = prev && wav.length ? pausePadSamples(prev, wav, this.sampleRate, pause) : 0;
      if (wav.length) prev = wav;
      yield { index: i, total: chunks.length, chunk: chunks[i], wav, padBefore };
    }
  }

  /** Whole utterance, joined exactly like join_audio_chunks(silence_ps=gaps_to_silence(gaps)). */
  async infer(text, opts = {}) {
    const wavs = [];
    for await (const r of this.inferStream(text, opts)) wavs.push(r.wav);
    const { gaps } = this.prepare(text, opts.maxChars ?? 140);
    return joinChunks(wavs, this.sampleRate, gaps.map((g) => GAP_SILENCE[g] ?? GAP_SILENCE.sentence));
  }
}

/** Load voices_v3_nano.json + .f32 (exported from vieneu/assets/voices_v3_nano.json). */
export async function loadVoices(jsonBuf, f32Buf) {
  const meta = JSON.parse(new TextDecoder().decode(jsonBuf));
  const all = new Float32Array(f32Buf);
  const list = meta.voices.map((m) => {
    const ss = m.style[0] * m.style[1];
    return { name: m.name, description: m.description, gender: m.gender,
      spk: all.slice(m.offset, m.offset + m.spk), style: all.slice(m.offset + m.spk, m.offset + m.spk + ss), styleShape: m.style };
  });
  return { default_voice: meta.default_voice, list };
}

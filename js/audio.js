// Port of vieneu_utils.core_utils audio helpers (edge_silence, trim_and_fade,
// pause_pad_samples, join_audio_chunks) + a WAV encoder.

export const EDGE_THRESH_DB = -45.0;
export const EDGE_KEEP_S = 0.04;
export const EDGE_FADE_S = 0.015;

/** [lead, tail] silent samples at each end (mean|x| over 10 ms windows). */
export function edgeSilence(wav, sr, threshDb = EDGE_THRESH_DB, winS = 0.01) {
  const n = wav.length;
  const win = Math.max(1, Math.trunc(winS * sr));
  const nWin = Math.floor(n / win);
  if (nWin === 0) return [n, 0];
  const th = 10 ** (threshDb / 20);
  let first = -1, last = -1;
  for (let w = 0; w < nWin; w++) {
    let s = 0;
    for (let k = w * win, e = k + win; k < e; k++) s += Math.abs(wav[k]);
    if (s / win > th) { if (first < 0) first = w; last = w; }
  }
  if (first < 0) return [n, 0];
  return [first * win, n - (last + 1) * win];
}

export function trimAndFade(wav, sr, threshDb = EDGE_THRESH_DB, keepS = EDGE_KEEP_S, fadeS = EDGE_FADE_S) {
  if (!wav.length) return wav;
  const [lead, tail] = edgeSilence(wav, sr, threshDb);
  const keep = Math.trunc(keepS * sr);
  const a = Math.max(0, lead - keep);
  const b = wav.length - Math.max(0, tail - keep);
  const out = Float32Array.from(wav.subarray(a, Math.max(a, b)));
  const n = Math.min(Math.trunc(fadeS * sr), Math.floor(out.length / 2));
  if (n > 0) {
    for (let k = 0; k < n; k++) {
      const r = n === 1 ? 0 : 0.5 - 0.5 * Math.cos((Math.PI * k) / (n - 1));   // np.linspace(0, pi, n)
      out[k] *= r;
      out[out.length - 1 - k] *= r;
    }
  }
  return out;
}

export function pausePadSamples(prev, next, sr, pauseS) {
  let [leadPrev, tail] = edgeSilence(prev, sr);
  if (leadPrev === prev.length) tail = prev.length;
  const [lead] = edgeSilence(next, sr);
  return Math.max(0, Math.trunc(pauseS * sr) - tail - lead);
}

export function joinChunks(chunks, sr, silencePs) {
  if (!chunks.length) return new Float32Array(0);
  const parts = [chunks[0]];
  for (let i = 1; i < chunks.length; i++) {
    const p = i - 1 < silencePs.length ? silencePs[i - 1] : 0;
    const pad = pausePadSamples(chunks[i - 1], chunks[i], sr, p);
    if (pad > 0) parts.push(new Float32Array(pad));
    parts.push(chunks[i]);
  }
  const out = new Float32Array(parts.reduce((s, x) => s + x.length, 0));
  let o = 0;
  for (const x of parts) { out.set(x, o); o += x.length; }
  return out;
}

/** 16-bit PCM mono WAV (same as soundfile PCM_16 used by tts.save). */
export function encodeWav(wav, sr) {
  const buf = new ArrayBuffer(44 + wav.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + wav.length * 2, true); w(8, "WAVE");
  w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, wav.length * 2, true);
  for (let i = 0; i < wav.length; i++) {
    const s = Math.max(-1, Math.min(1, wav[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

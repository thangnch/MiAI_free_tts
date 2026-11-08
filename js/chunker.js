// Port of vieneu_utils (phonemize_text.py + core_utils.py) — text → chunks → phonemes.
// Mirrors the Python v3 Nano path: normalize_to_chunks_v3_with_gaps(max_chars=140)
// then phonemize_text_with_emotions(chunk) for every chunk.
// `g2p` is a SeaG2p instance (WASM) exposing normalize(text, puncNorm) / phonemize(text, puncNorm),
// `puncNorm` is the sea-g2p trailing-punctuation rule (also from WASM).

const clen = (s) => { let n = 0; for (const _ of s) n++; return n; };   // Python len() = code points
const pyStrip = (s) => s.replace(/^\s+|\s+$/gu, "");
const isSpace = (ch) => /^\s$/u.test(ch);

// ── emotion cues ─────────────────────────────────────────────────────────────
const EMOTION_TAG_TO_K = {
  "chuckle": 1, "cười": 1, "cuoi": 1,
  "sigh": 2, "thở dài": 2, "tho dai": 2,
  "clear throat": 3, "hắng giọng": 3, "hang giong": 3,
};
const EMOTION_SPLIT_RE = /(\[[^\]]+\]|<\|emotion_\d+\|>)/u;
const ATTACHING_PUNCT = new Set([...".,!?;:…)]}\"'’”"]);
const EMOTION_TOKEN_RE = /<\|emotion_\d+\|>/gu;

function emotionTagToken(tag) {
  const t = pyStrip(tag);
  if (t.startsWith("<|")) return t;
  const inner = pyStrip(t.slice(1, -1)).toLowerCase();
  const k = EMOTION_TAG_TO_K[inner];
  return k !== undefined ? `<|emotion_${k}|>` : null;
}

// ── sentence splitting (quote/bracket aware) ─────────────────────────────────
const OPEN_TO_CLOSE = { "(": ")", "[": "]", "{": "}", "“": "”", "‘": "’", "«": "»", "‹": "›", "「": "」", "『": "』" };
const OPENERS = new Set(Object.keys(OPEN_TO_CLOSE));
const CLOSERS = new Set(Object.values(OPEN_TO_CLOSE));
const SYMMETRIC_QUOTE = '"';
const SENT_END = new Set([...".!?…"]);
const TRAILING_CLOSE = new Set([...CLOSERS, ..."\"'’”"]);

function scanSentences(text, quoteAware) {
  const c = Array.from(text);
  const n = c.length;
  const out = [];
  let start = 0, i = 0, depth = 0, inQuote = false;
  while (i < n) {
    const ch = c[i];
    if (quoteAware && ch === SYMMETRIC_QUOTE) inQuote = !inQuote;
    else if (quoteAware && OPENERS.has(ch)) depth++;
    else if (quoteAware && CLOSERS.has(ch)) { if (depth) depth--; }
    else if (SENT_END.has(ch) && depth === 0 && !inQuote) {
      let j = i + 1;
      while (j < n && SENT_END.has(c[j])) j++;
      while (j < n && TRAILING_CLOSE.has(c[j])) j++;
      if (j >= n || isSpace(c[j])) {
        out.push(c.slice(start, j).join(""));
        start = i = j;
        continue;
      }
      i = j;
      continue;
    }
    i++;
  }
  if (start < n) out.push(c.slice(start).join(""));
  return [out.map(pyStrip).filter(Boolean), depth === 0 && !inQuote];
}

export function splitIntoSentences(text) {
  if (!text) return [];
  let [s, balanced] = scanSentences(text, true);
  if (!balanced) [s] = scanSentences(text, false);
  return s;
}

// ── packing sentences into chunks ────────────────────────────────────────────
const CONN_WORDS = new Set("và nhưng hoặc song rồi nên vì nếu khi để do bởi".split(" "));
const CONN_PAIRS = new Set([
  ["sau", "khi"], ["trước", "khi"], ["trong", "khi"], ["mỗi", "khi"],
  ["đến", "khi"], ["tới", "khi"], ["cho", "nên"], ["cho", "đến"],
  ["bởi", "vì"], ["nếu", "như"], ["tuy", "nhiên"], ["thế", "nhưng"],
  ["vì", "vậy"], ["vì", "thế"], ["do", "đó"], ["sau", "đó"],
].map(([a, b]) => a + "\u0000" + b));
const isPair = (a, b) => CONN_PAIRS.has(a + "\u0000" + b);
const CONN_STRIP = new Set([..."\"'“”‘’()[]«»…"]);
const NUMBER_WORDS = new Set(("không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi trăm nghìn ngàn " +
  "triệu tỷ tỉ linh lẻ phẩy chấm").split(" ").filter(Boolean));
const CHUNK_TAIL_SLACK = 15;
const RE_MINOR_PUNCT = /(?<=[,;:\-–—])\s+/u;
const RE_TOKEN_KEEP_EN = /<en>.*?<\/en>|\S+/gisu;

function connKey(tok) {
  const c = Array.from(tok);
  let a = 0, b = c.length;
  while (a < b && CONN_STRIP.has(c[a])) a++;
  while (b > a && CONN_STRIP.has(c[b - 1])) b--;
  return c.slice(a, b).join("").toLowerCase();
}
const isNumberWord = (t) => { const k = connKey(t); return NUMBER_WORDS.has(k) || /^\p{Nd}+$/u.test(k); };
const tailSlack = (m) => Math.min(CHUNK_TAIL_SLACK, Math.floor(m / 8));
function fits(cur, add, m) {
  const total = cur ? cur + 1 + add : add;
  const slack = tailSlack(m);
  return total <= m || (add <= slack && total <= m + slack);
}
function spanLen(words, s, e) {
  if (e <= s) return 0;
  let n = e - s - 1;
  for (let k = s; k < e; k++) n += clen(words[k]);
  return n;
}

function balancedCut(words, start, target, maxChars, minLeft) {
  let bestNat = null, bestPlain = null, endCap = start + 1;
  for (let j = start + 1; j < words.length; j++) {
    const left = spanLen(words, start, j);
    if (left > maxChars) break;
    endCap = j;
    const key = connKey(words[j]), prev = connKey(words[j - 1]);
    const nxt = j + 1 < words.length ? connKey(words[j + 1]) : "";
    const d = Math.abs(left - target);
    const insidePair = isPair(prev, key);
    const natural = (CONN_WORDS.has(key) || isPair(key, nxt)) && !CONN_WORDS.has(prev);
    if (natural && !insidePair && left >= minLeft && (bestNat === null || d < bestNat[0])) bestNat = [d, j];
    const plainOk = !insidePair && !(isNumberWord(words[j - 1]) && isNumberWord(words[j]));
    if (plainOk && (bestPlain === null || d < bestPlain[0])) bestPlain = [d, j];
  }
  if (bestNat) return bestNat[1];
  if (bestPlain) return bestPlain[1];
  let end = endCap;
  while (end > start + 1 && isPair(connKey(words[end - 1]), connKey(words[end]))) end--;
  return end;
}

function splitLongPart(part, maxChars) {
  const words = part.match(RE_TOKEN_KEEP_EN) || [];
  const minLeft = Math.floor(maxChars / 3);
  const pieces = [];
  let start = 0;
  while (start < words.length) {
    const rest = spanLen(words, start, words.length);
    if (rest <= maxChars) { pieces.push(words.slice(start).join(" ")); break; }
    let full = start + 1;
    while (full < words.length && spanLen(words, start, full + 1) <= maxChars) full++;
    if (fits(spanLen(words, start, full), spanLen(words, full, words.length), maxChars)) {
      pieces.push(words.slice(start).join(" ")); break;
    }
    const nPieces = Math.ceil(rest / maxChars);
    const end = balancedCut(words, start, rest / nPieces, maxChars, minLeft);
    pieces.push(words.slice(start, end).join(" "));
    start = end;
  }
  return pieces;
}

export function packSentencesIntoChunks(sentences, maxChars = 256) {
  const out = [];
  let buf = "";
  for (let sentence of sentences) {
    sentence = pyStrip(sentence);
    if (!sentence) continue;
    if (clen(sentence) > maxChars) {
      if (buf) { out.push(buf); buf = ""; }
      for (let part of sentence.split(RE_MINOR_PUNCT)) {
        part = pyStrip(part);
        if (!part) continue;
        if (fits(clen(buf), clen(part), maxChars)) buf = buf ? buf + " " + part : part;
        else {
          if (buf) out.push(buf);
          buf = part;
          if (clen(buf) > maxChars) {
            const pieces = splitLongPart(buf, maxChars);
            out.push(...pieces.slice(0, -1));
            buf = pieces.length ? pieces[pieces.length - 1] : "";
          }
        }
      }
    } else if (buf && !fits(clen(buf), clen(sentence), maxChars)) {
      out.push(buf); buf = sentence;
    } else {
      buf = buf ? buf + " " + sentence : sentence;
    }
  }
  if (buf) out.push(buf);
  return out.map(pyStrip).filter(Boolean);
}

function classifyGap(chunk) {
  const c = chunk.replace(/\s+$/u, "");
  return c && ".!?".includes(c[c.length - 1]) ? "sentence" : "minor";
}

// ── normalisation by paragraph / sentence ────────────────────────────────────
function normalizeSentenceKeepCues(sentence, g2p) {
  if (!sentence.includes("[") && !sentence.includes("<|emotion_")) return g2p.normalize(sentence, false);
  const parts = [];
  sentence.split(EMOTION_SPLIT_RE).forEach((part, i) => {
    if (i % 2 === 1) { const tok = emotionTagToken(part); parts.push(tok !== null ? tok : part); }
    else if (pyStrip(part)) parts.push(g2p.normalize(part, false));
  });
  return parts.filter(Boolean).join(" ");
}

function normalizedSentencesByPara(text, g2p, keepCues) {
  const paras = text.split(/[\r\n]+/u).filter((p) => pyStrip(p));
  const out = [];
  for (const para of paras) {
    const sents = splitIntoSentences(para);
    if (!sents.length) continue;
    out.push(sents.map((s) => (keepCues ? normalizeSentenceKeepCues(s, g2p) : g2p.normalize(s, false))));
  }
  return out;
}

const effectiveLen = (c) => clen(pyStrip(c.replace(EMOTION_TOKEN_RE, "")));

function mergeShortChunks(chunks, gaps, minChars) {
  while (chunks.length > 1) {
    const short = [];
    chunks.forEach((c, i) => { if (effectiveLen(c) < minChars) short.push(i); });
    if (!short.length) break;
    let i = short[0];
    for (const k of short) if (effectiveLen(chunks[k]) < effectiveLen(chunks[i])) i = k;
    const sides = [];
    if (i < chunks.length - 1) sides.push([[gaps[i] !== "para" ? 1 : 0, -clen(chunks[i + 1])], "R"]);
    if (i > 0) sides.push([[gaps[i - 1] !== "para" ? 1 : 0, -clen(chunks[i - 1])], "L"]);
    let best = sides[0];
    for (const s of sides.slice(1)) {
      if (s[0][0] > best[0][0] || (s[0][0] === best[0][0] && s[0][1] > best[0][1])) best = s;
    }
    if (best[1] === "R") { chunks[i] = chunks[i] + " " + chunks.splice(i + 1, 1)[0]; gaps.splice(i, 1); }
    else { chunks[i - 1] = chunks[i - 1] + " " + chunks.splice(i, 1)[0]; gaps.splice(i - 1, 1); }
  }
  return [chunks, gaps];
}

/** normalize_to_chunks_v3_with_gaps → [chunks, gaps] */
export function normalizeToChunks(text, g2p, puncNorm, maxChars = 140, minChunkChars = 20) {
  if (!text) return [[], []];
  const keepCues = text.includes("[") || text.includes("<|emotion_");
  let chunks = [], gaps = [];
  for (const sents of normalizedSentencesByPara(text, g2p, keepCues)) {
    const pc = packSentencesIntoChunks(sents, maxChars);
    if (!pc.length) continue;
    if (chunks.length) gaps.push("para");
    pc.forEach((ch, j) => { if (j > 0) gaps.push("sentence"); chunks.push(ch); });
  }
  chunks = chunks.map((c) => puncNorm(c));
  gaps = gaps.map((g, i) => (g === "para" ? g : classifyGap(chunks[i])));
  return mergeShortChunks(chunks, gaps, minChunkChars);
}

/** phonemize_text_with_emotions */
export function phonemizeWithEmotions(text, g2p, puncNorm) {
  if (!text.includes("[") && !text.includes("<|emotion_")) return g2p.phonemize(text, true);
  let out = "";
  text.split(EMOTION_SPLIT_RE).forEach((part, i) => {
    const token = i % 2 === 1 ? emotionTagToken(part) : null;
    if (token !== null) { out = out ? out + " " + token : token; return; }
    const ph = part && pyStrip(part) ? g2p.phonemize(part, false) : "";
    if (!ph) return;
    if (!out) out = ph;
    else if (ATTACHING_PUNCT.has(Array.from(ph)[0])) out += ph;
    else out += " " + ph;
  });
  return puncNorm(out);
}

export const GAP_SILENCE = { para: 0.70, sentence: 0.50, minor: 0.30 };

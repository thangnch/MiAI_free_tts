// Minimal .npz reader (zip of .npy files; stored or deflated). Float32/float64/int arrays only.

async function inflateRaw(bytes) {
  const ds = new DecompressionStream("deflate-raw");
  const out = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(out);
}

function parseNpy(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const major = u8[6];
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(u8.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  if (/'fortran_order':\s*True/.test(header)) throw new Error("fortran_order npy not supported");
  const body = u8.slice(hstart + hlen);   // copy → aligned
  const types = { "<f4": Float32Array, "<f8": Float64Array, "<i4": Int32Array, "<i8": BigInt64Array, "|b1": Uint8Array, "|u1": Uint8Array };
  const T = types[descr];
  if (!T) throw new Error("unsupported npy dtype " + descr);
  let data = new T(body.buffer, 0, body.byteLength / T.BYTES_PER_ELEMENT);
  if (T === Float64Array) data = Float32Array.from(data);
  return { data, shape, dtype: descr };
}

export async function loadNpz(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  let eocd = -1;
  for (let i = u8.length - 22; i >= Math.max(0, u8.length - 65557); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("not a zip/npz");
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const out = {};
  for (let k = 0; k < count; k++) {
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true), xlen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true);
    let lho = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(u8.subarray(p + 46, p + 46 + nlen));
    if (csize === 0xffffffff || lho === 0xffffffff) {           // zip64 extra field
      let q = p + 46 + nlen; const end = q + xlen;
      while (q < end) {
        const id = dv.getUint16(q, true), sz = dv.getUint16(q + 2, true);
        if (id === 1) {
          let r = q + 4;
          if (dv.getUint32(p + 24, true) === 0xffffffff) r += 8;
          if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(r, true)); r += 8; }
          if (lho === 0xffffffff) lho = Number(dv.getBigUint64(r, true));
        }
        q += 4 + sz;
      }
    }
    const lnlen = dv.getUint16(lho + 26, true), lxlen = dv.getUint16(lho + 28, true);
    const start = lho + 30 + lnlen + lxlen;
    let raw = u8.subarray(start, start + csize);
    if (method === 8) raw = await inflateRaw(raw);
    else if (method !== 0) throw new Error("unsupported zip method " + method);
    out[name.replace(/\.npy$/, "")] = parseNpy(raw);
    p += 46 + nlen + xlen + clen;
  }
  return out;
}

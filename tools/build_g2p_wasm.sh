#!/usr/bin/env bash
# Rebuild g2p/ (sea-g2p → WebAssembly). Needs: rust (stable is fine), wasm-bindgen-cli 0.2.100, python3.
# Uses -Zbuild-std so no prebuilt wasm32 std is required (RUSTC_BOOTSTRAP enables it on stable).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${OUT:-$HERE/../g2p}"; WORK="${WORK:-/tmp/sea-g2p-wasm}"
SEA_COMMIT=e825173f235d08ea19315b2b279fb11153b44cea     # sea-g2p v0.10.0
rm -rf "$WORK" && git clone -q https://github.com/pnnbao97/sea-g2p "$WORK" && cd "$WORK" && git checkout -q $SEA_COMMIT
git apply "$HERE/sea-g2p-wasm.patch"
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then BS=(); else
  BS=(-Zbuild-std=std,panic_abort); export RUSTC_BOOTSTRAP=1
  [ -d "$(rustc --print sysroot)/lib/rustlib/src/rust/library" ] || rustup component add rust-src
fi
cargo build --release --lib --target wasm32-unknown-unknown "${BS[@]}" --no-default-features --features wasm
wasm-bindgen target/wasm32-unknown-unknown/release/sea_g2p_rs.wasm --target web --out-dir "$OUT" --out-name sea_g2p
rm -f "$OUT"/*.d.ts
# dictionary: take sea_g2p.bin from the matching PyPI wheel, drop the Thai/Indonesian sections
pip download -q --no-deps sea-g2p==0.10.0 -d "$WORK/whl" && (cd "$WORK/whl" && unzip -qo ./*.whl 'sea_g2p/sea_g2p.bin')
python3 "$HERE/strip_dict.py" "$WORK/whl/sea_g2p/sea_g2p.bin" "$WORK/scripts" "$WORK/sea_g2p_vi.bin"
gzip -9 -c "$WORK/sea_g2p_vi.bin" > "$OUT/sea_g2p_vi.bin.gz"
echo "done → $OUT"

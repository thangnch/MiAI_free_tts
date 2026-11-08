# VieNeu-TTS v3 Nano chạy trên trình duyệt (WebGPU)

Trang tĩnh, toàn bộ xử lý chạy **trên máy người dùng**: chuẩn hoá văn bản + phiên âm (sea-g2p biên dịch sang WebAssembly) và 4 model ONNX của Nano (onnxruntime-web, WebGPU; tự lùi về CPU/WASM nếu trình duyệt không có WebGPU). Server chỉ phát file tĩnh, không tốn CPU/GPU.


## Chạy thử

```bash
cd vieneu-web
python3 -m http.server 8000      # mở http://localhost:8000
```

Phải mở qua `http://localhost` hoặc `https://` (không mở trực tiếp file://). Lần đầu trình duyệt tải:

| Thành phần | Nguồn | Dung lượng |
|---|---|---|
| Từ điển phiên âm `g2p/sea_g2p_vi.bin.gz` | server của bạn | 15 MB (giải nén 50 MB trong RAM) |
| `g2p/sea_g2p_bg.wasm` | server của bạn | 3,7 MB |
| Giọng preset `voices/*` | server của bạn | 0,6 MB |
| 4 model ONNX + config | Hugging Face `pnnbao-ump/VieNeu-TTS-v3-Nano` | tuỳ model |
| onnxruntime-web 1.31.1 | cdn.jsdelivr.net | 29 MB wasm (CDN nén lại khi truyền) |

Từ điển và model được lưu vào Cache Storage, các lần sau không tải lại.

## Tham số URL

- `?models=./models`: lấy model từ thư mục của bạn thay vì Hugging Face (copy `text_encoder.onnx`, `duration_predictor.onnx`, `vector_estimator.onnx`, `codec_decoder.onnx`, `config.json`, `constants.npz` vào `models/`). Nên dùng khi triển khai nội bộ.
- `?ep=wasm`: ép chạy CPU để so sánh tốc độ.
- `?voice=Mỹ Duyên`: giọng mặc định.

## Triển khai

Bất kỳ static host nào (nginx, S3/CloudFront, GitHub Pages…). Gợi ý nginx:

```nginx
types { application/wasm wasm; }
location ~* \.(wasm|onnx|gz|f32)$ { add_header Cache-Control "public, max-age=31536000, immutable"; }
```

Không bật `gzip` cho `sea_g2p_vi.bin.gz` (đã nén sẵn, trang tự giải nén bằng `DecompressionStream`).

Tuỳ chọn: thêm header `Cross-Origin-Opener-Policy: same-origin` và `Cross-Origin-Embedder-Policy: credentialless` để chế độ CPU dùng được đa luồng.

## Cấu trúc

```
index.html          giao diện
js/app.js           khởi tạo, chọn WebGPU/WASM, UI, phát từng đoạn
js/tts.js           API giống Vieneu.infer / infer_stream
js/nano.js          port OnnxV3NanoEngine (Euler + CFG, duration, decoder)
js/chunker.js       port normalize_to_chunks_v3_with_gaps + phonemize_text_with_emotions
js/audio.js         trim_and_fade, join_audio_chunks, xuất WAV
js/npz.js           đọc constants.npz
g2p/                sea-g2p 0.10.0 → WebAssembly + từ điển (đã bỏ phần tiếng Thái/Indonesia)
voices/             11 giọng preset của Nano (từ vieneu/assets/voices_v3_nano.json)
tools/              patch + script build lại g2p/
```


## Build lại `g2p/`

```bash
cargo install wasm-bindgen-cli --version 0.2.100
./tools/build_g2p_wasm.sh
```

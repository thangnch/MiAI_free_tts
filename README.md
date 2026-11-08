# VieNeu-TTS v3 Nano chạy trên trình duyệt (WebGPU)

Trang tĩnh, toàn bộ xử lý chạy **trên máy người dùng**: chuẩn hoá văn bản + phiên âm (sea-g2p biên dịch sang WebAssembly) và 4 model ONNX của Nano (onnxruntime-web, WebGPU; tự lùi về CPU/WASM nếu trình duyệt không có WebGPU). Server chỉ phát file tĩnh, không tốn CPU/GPU.

Tương đương đoạn Python:

```python
tts = Vieneu(mode="v3nano")
audio = tts.infer(text, voice="Mạnh Dũng", speed=0.8)
```

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

## Đã kiểm chứng

- Phiên âm WASM so với `sea_g2p` Python: giống hệt từng ký tự (cả số, ngày tháng, URL, tiếng Anh xen kẽ, emotion cue).
- Chia đoạn + khoảng nghỉ so với `vieneu_utils`: giống hệt trên 8 văn bản mẫu.
- Toàn bộ pipeline JS so với `OnnxV3NanoEngine` Python (cùng noise, model giả lập cùng chữ ký input): 24/24 trường hợp, sai số tối đa 3,6·10⁻⁷.
- Chạy end-to-end trong Chrome headless ở cả WebGPU và WASM.

Chưa kiểm chứng với file model Nano thật (không tải được từ môi trường build). Nếu một op nào đó chưa có kernel WebGPU, onnxruntime tự chạy op đó trên CPU; nếu tạo session WebGPU lỗi hẳn, trang tự chuyển sang WASM.

## Khác biệt so với bản Python

- Noise ngẫu nhiên dùng bộ sinh riêng (mulberry32 + Box–Muller), nên cùng `seed` vẫn cho kết quả khác Python, nhưng lặp lại được trên web.
- Không có watermark Perth (bản Python cũng tắt nếu chưa cài `resemble-perth`).
- Chưa có clone giọng (Nano cần thêm 3 model encoder), chỉ dùng 11 giọng preset.

## Build lại `g2p/`

```bash
cargo install wasm-bindgen-cli --version 0.2.100
./tools/build_g2p_wasm.sh
```

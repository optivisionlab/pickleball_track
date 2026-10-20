# benchmark_vlm: VLM zero-shot trên NSVA (made / miss / no-event)

Làm theo `Guide.md` + `eval_vlm.py` gốc, chạy nhiều model đa phương thức không fine-tune và so với custom model (v3/v4/v5) trên **cùng 5 fold test**.

```bash
./run_models.sh                         # tất cả model, họ Qwen trước; model đã xong thì bỏ qua
./run_models.sh qwen3-vl-8b gemma4-e4b  # chỉ vài model
STAGES=smoke ./run_models.sh            # chỉ chạy thử 10 clip mỗi model
```

Kết quả so sánh: **`runs/model_comparison.md`** (và `.csv`), cập nhật sau mỗi model. Có thể tạo lại bất cứ lúc nào bằng `python eval_vlm.py summary`.

## Model

| Thứ tự | Tên (`runs/<tên>/`) | Model | Nhóm | Backend | Env |
|---|---|---|---|---|---|
| 1 | qwen3.5-4b | Qwen/Qwen3.5-4B | ~4B | qwen | score-video (đã chạy, giữ nguyên kết quả) |
| 2 | qwen3-vl-4b | Qwen/Qwen3-VL-4B-Instruct | ~4B | qwen | score-video |
| 3 | qwen3-vl-8b | Qwen/Qwen3-VL-8B-Instruct | ~8B | qwen | score-video |
| 4 | qwen3.5-9b | Qwen/Qwen3.5-9B | ~8B | qwen | score-video |
| 5 | internvl3.5-4b | OpenGVLab/InternVL3_5-4B-HF | ~4B | hf | score-video |
| 6 | gemma4-e4b | google/gemma-4-E4B-it | ~4B | hf | score-video |
| 7 | minicpm-v-4 | openbmb/MiniCPM-V-4 | ~4B | minicpm | vlm-tf4 |
| 8 | internvl3.5-8b | OpenGVLab/InternVL3_5-8B-HF | ~8B | hf | score-video |
| 9 | minicpm-v-4.5 | openbmb/MiniCPM-V-4_5 (`--pack 3`) | ~8B | minicpm | vlm-tf4 |
| 10 | keye-vl-1.5-8b | Kwai-Keye/Keye-VL-1_5-8B | ~8B | keye | vlm-tf4 |

Danh sách, batch size và VRAM cần cho từng model nằm ở đầu `run_models.sh` (`MODELS=(...)`).

- **Hai môi trường**: Qwen3.5 / Gemma 4 cần transformers 5.x (`score-video`). MiniCPM-V và Keye là remote code viết cho transformers 4.x: trên 5.x MiniCPM-V vẫn load được nhưng phần vision cho ra rác (mô tả ảnh bóng rổ thành "các vạch đỏ trắng"), nên chạy trong env `vlm-tf4` (transformers 4.51.3, xem `requirements-tf4.txt`). Keye bắt buộc có `flash-attn`.
- **Cách chấm giống nhau cho mọi model**: cùng prompt A/B/C, đọc xác suất 3 chữ cái ở token đầu tiên. Mỗi chữ cái cộng xác suất mọi cách viết 1 token của nó (`A`, `▁A`, `ĠA`): tokenizer SentencePiece (MiniCPM-V 4, Gemma) mã hoá "A" thành `▁A` trong khi model trả lời bằng `A`. MiniCPM-V dùng API `chat()` riêng nên logit lấy qua hook ở `lm_head`.
- **MiniCPM-V 4.5 `--pack 3`**: mỗi nhóm 3 frame liên tiếp được nén chung (3D resampler, kèm `temporal_ids` theo 0.1 s), nên với cấu hình 16 frame model thực sự xem 48 frame (~5 fps trên clip 9 s) mà tốn token như 16 frame.
- **Sửa tương thích** (trong `vlm_backends.py`): InternVL-HF ép video về 448×448 (config video ghi nhầm 384); Gemma 4 bơm `num_frames_per_video` mà transformers 5.20 quên truyền; Gemma 4 dùng attention mặc định vì phần audio không có sdpa.

## Mỗi model chạy những gì

| Bước Guide | Stage | Ghi chú |
|---|---|---|
| – | `batch` | `find_batch.py`: đo batch lớn nhất vừa A30 (giữ trống 1.5 GB) với cấu hình nặng nhất, lưu `batch_sizes.json` (đo 1 lần mỗi model) |
| 3 | `smoke` | 10 clip val; dừng model nếu `mean_letter_mass` < 0.3 (model không trả lời bằng chữ cái) hoặc lỗi nhiều |
| 4 | `val` | 3 cấu hình (16f/448, 32f/448, 16f/640) trên `val_fold1.csv`, chọn macro-F1 cao nhất → `runs/<tên>/best_config.txt` |
| 5 | `test` | 1 lần trên `splits/test.csv` (8329 clip), `--resume` nếu bị dừng |
| 6 | `compare` | McNemar với v3/v4/v5 → `runs/<tên>/test_final/comparison_custom_preds_v*.json` |

- Trước mỗi model script chờ đến khi GPU đủ VRAM trống (server dùng chung).
- Model không vừa GPU (kể cả batch 1) hoặc không qua smoke thì bị **bỏ qua** (`STATUS` = `skipped: ...` / `failed: ...`) và chạy tiếp model sau. `CPU_FALLBACK=1` chạy model không vừa GPU trên CPU (rất chậm, ~1 phút/clip).
- Kiểm tra nhanh mọi model (1 clip): `STAGES=smoke SMOKE_LIMIT=1 ./run_models.sh`.
- Model lỗi được ghi vào `runs/<tên>/STATUS` và script chạy tiếp model sau. Chạy lại `./run_models.sh <tên>` để tiếp tục.
- `SWEEP=0` bỏ bước val, dùng cấu hình cố định `FIXED_CONFIG` (mặc định 16f/640) để tiết kiệm thời gian.
- Log: `logs/<tên>.log`.

## Chia dữ liệu (giống hệt custom model v3/v4/v5)

`make_splits.py` chép lại logic của `baseline/benchmark_common.py` (nhánh `baseline_shot1`):

- 8329 clip sạch của file mapping 1-1, 96 trận.
- 5 fold `StratifiedGroupKFold` theo `game_id`, seed 42 → mỗi fold có tập **test** riêng; 5 tập test ghép lại = toàn bộ 8329 clip (`splits/test.csv`, có cột `fold`).
- Trong mỗi fold, **val** = split đầu tiên của `StratifiedGroupKFold(8)` trên phần train (custom model dùng để early-stop).
- Script tự kiểm tra danh sách trận test của từng fold khớp với `fold_k/result.json` của custom model.
- Nhãn đổi tên: `hit → made`, `no_event → no-event`.

| fold | val | test |
|---|---|---|
| 1 | 802 | 1693 |
| 2 | 823 | 1697 |
| 3 | 811 | 1672 |
| 4 | 806 | 1629 |
| 5 | 820 | 1638 |

`export_custom_preds.py` xuất dự đoán từng clip của v3/v4/v5 từ `best.pt` mỗi fold (khớp số đã lưu: v3 0.8195 ± 0.0045).

Lưu ý khi đọc kết quả: clip của `val_fold1` nằm trong tập test của fold 2–5. Model không được train, chỉ chọn 1 trong 3 cấu hình, nên ảnh hưởng nhỏ; con số sạch hoàn toàn là cột `fold1_f1`.

## Bảng so sánh (`runs/model_comparison.md`)

Mỗi dòng một model (kể cả custom v3/v4/v5): macro-F1 + CI 95%, mean ± std theo fold, F1 fold 1, accuracy, F1 từng lớp, AUC, p-value McNemar so với v3, `mean_letter_mass`, tốc độ (clip/s), VRAM đỉnh, cấu hình được chọn và trạng thái (`done` / `failed: ...` / `not run`).

## Tốc độ / batch

- `--batch-size` clip mỗi lần forward và `--workers` tiến trình CPU (mặc định 16, mỗi tiến trình 1 thread) giải mã video trước cho GPU. Giải mã video (~1.2 s/clip) là nút cổ chai chính.
- Nếu batch bị OOM, batch size tự giảm một nửa cho phần còn lại của lần chạy.
- `BATCH_SCALE=2 ./run_models.sh` nhân đôi batch của mọi model khi GPU rảnh.
- Kiểm tra model không cần GPU: `python eval_vlm.py run ... --device cpu --limit 2 --workers 0`.

## Ghi chú

- Qwen3.5 chạy kernel PyTorch dự phòng cho các lớp linear-attention (thiếu `causal_conv1d` / `flash-linear-attention`): kết quả đúng, chỉ chậm hơn.
- Qwen3.5-4B zero-shot đoán gần như mọi clip là `made` (macro-F1 0.24).

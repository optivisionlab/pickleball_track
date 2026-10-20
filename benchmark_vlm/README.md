# benchmark_vlm: Qwen3.5-4B zero-shot trên NSVA (made / miss / no-event)

Làm theo `Guide.md` + `eval_vlm.py` gốc, chỉ thay model bằng `Qwen/Qwen3.5-4B`.
Môi trường: conda `score-video` (cài thêm bằng `requirements.txt`).

```bash
./run_benchmark.sh            # chạy toàn bộ: splits -> smoke -> val -> test -> compare
./run_benchmark.sh val        # hoặc từng bước
```

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

## Các bước

| Bước Guide | Stage | Ghi chú |
|---|---|---|
| 3 | `smoke` | 10 clip val; kiểm tra `error`, `mean_letter_mass` ≈ 1, `peak_vram_gb` |
| 4 | `val` | 3 cấu hình (16f/448, 32f/448, 16f/640) trên `val_fold1.csv`, chọn macro-F1 cao nhất → `runs/best_config.txt` |
| 5 | `test` | chạy 1 lần trên `splits/test.csv` (8329 clip, ~4 giờ), `--resume` nếu bị dừng |
| 6 | `compare` | `export_custom_preds.py` xuất dự đoán từng clip của v3/v4/v5 từ `best.pt` mỗi fold, rồi McNemar |

`metrics.json` của bước test có thêm `per_fold` và `fold_macro_f1_mean/std`, so trực tiếp với `kfold_summary.json` của custom model (v3 0.8195 ± 0.0045).

Lưu ý khi đọc kết quả: clip của `val_fold1` nằm trong tập test của fold 2–5. Model không được train, chỉ chọn 1 trong 3 cấu hình, nên ảnh hưởng nhỏ; con số sạch hoàn toàn là `per_fold["1"]`.

## Kết quả smoke (10 clip, 16f/448)

- Không lỗi, `mean_letter_mass` = 0.9999, VRAM đỉnh 9.7 GB (1 clip/lần) trên A30.
- Model nhìn thấy video (mô tả đúng cảnh), nhưng zero-shot có xu hướng trả lời gần như mọi clip là `made`.

## Tốc độ / batch

- `eval_vlm.py run` xử lý `--batch-size` clip mỗi lần forward (mặc định 8) và dùng `--workers` tiến trình CPU (mặc định 16, mỗi tiến trình 1 thread) để giải mã video trước cho GPU.
- Giải mã video (~1.2 s/clip với decord) là nút cổ chai chính, không phải GPU. Batch 8: VRAM đỉnh ~13.7 GB, ~1.3–1.5 clip/s (bản chạy từng clip: ~0.58 clip/s). Batch 16: ~14.7 GB nhưng không nhanh hơn khi CPU bận.
- Kết quả batch giống chạy từng clip (64 clip: cùng nhãn dự đoán, chênh xác suất tối đa 0.03 do bf16).
- Đổi qua biến môi trường: `BATCH=16 WORKERS=24 ./run_benchmark.sh test`. Nếu batch bị OOM, script tự chạy lại từng clip.

## Ghi chú

- Qwen3.5 chạy kernel PyTorch dự phòng cho các lớp linear-attention (thiếu `causal_conv1d` / `flash-linear-attention`): kết quả đúng, chỉ chậm hơn.

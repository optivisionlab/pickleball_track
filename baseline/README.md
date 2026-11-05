# Model custom v3 / v4 / v5: kết quả, mô hình và cách train

Ba mô hình **v3 / v4 / v5** được đánh giá trên 4 bộ dữ liệu: **pickleball** (dữ liệu của nhóm), **bóng rổ NSVA**, **bóng đá SoccerNet shots** và **UCF-101**. Mỗi môn chỉ giữ **cấu hình tốt nhất của từng version**. Cấu hình này được chọn trên tập validation; tập test chỉ dùng để báo cáo.

Dữ liệu sinh ra nằm ngay trong thư mục của từng môn, và git bỏ qua chúng:
- `<môn>/cache/`: feature cache.
- `<môn>/outputs/runs/`: checkpoint và kết quả của cấu hình tốt nhất.

Encoder pretrained dùng chung cho v3/v4/v5 nằm trong `common/pretrained/` (git bỏ qua): checkpoint BEATs (base và fine-tune AudioSet), mã nguồn BEATs (`unilm`), VideoMAE-base (self-supervised, chỉ encoder). Các model Hugging Face còn lại (VideoMAE K400, CLIP, Whisper) được tải theo tên, từ cache HF của máy.

---

## 1. Kết quả tốt nhất

| Môn | Nhiệm vụ | Đánh giá | Chỉ số | **v3** | **v4** | **v5** |
|---|---|---|---|---|---|---|
| Pickleball | phát hiện cú đánh hit / miss theo thời gian trong video dài | 5-fold theo người chơi, 115 video, 17 người | event macro-F1 (±0,5 s) | 0,931 | **0,951** | 0,838 |
| Bóng rổ NSVA | phân loại clip made / miss / no_event | 5-fold theo trận, 8 329 clip, 96 trận; 3 seed | macro-F1 | **0,832** ± 0,004 | 0,828 ± 0,001 | 0,796 ± 0,001 |
| Bóng đá SoccerNet, 3 nhãn | goal / saved / off_target | train / valid / test cố định, test 2 137 clip; 5 seed | macro-F1 | **0,807** ± 0,002 | 0,807 ± 0,007 | 0,786 ± 0,005 |
| Bóng đá SoccerNet, 2 nhãn | goal / saved (saved = saved + off_target) | như trên | macro-F1 | **0,970** ± 0,002 | 0,968 ± 0,002 | 0,954 ± 0,004 |
| UCF-101 | nhận dạng 101 hành động | 3 split chính thức | top-1 (%) | **98,1** ± 0,7 (fine-tune end-to-end: 98,1 ± 0,6) | 97,8 ± 0,6 | 97,9 ± 0,5 |

| Accuracy | v3 | v4 | v5 |
|---|---|---|---|
| Pickleball (theo token) | 0,905 | 0,912 | 0,840 |
| NSVA | 0,833 | 0,829 | 0,796 |
| Bóng đá 3 nhãn | 0,779 | 0,775 | 0,753 |
| Bóng đá 2 nhãn (lớp saved chiếm 84,6%) | 0,984 | 0,983 | 0,976 |
| UCF-101 (= top-1) | 98,1 | 97,8 | 97,9 |

- Số "± x" là độ lệch chuẩn giữa các seed (NSVA, bóng đá) hoặc giữa các split (UCF-101).
- v3 và v4 gần như ngang nhau trên NSVA, bóng đá và UCF-101. Trên pickleball, v4 cao hơn v3 rõ rệt.
- v5 (không dùng audio) luôn thấp nhất, chênh nhiều nhất ở pickleball, nơi âm thanh chạm bóng là tín hiệu chính.

---

## 2. So sánh với VLM zero-shot

Nguồn: nhánh `llm_benchmark`, thư mục `benchmark_vlm/` (`runs/comparison.md` cho NSVA, `runs_pickleball/comparison.md` cho pickleball).

- VLM không được train. Cấu hình frame của VLM được chọn trên `val_fold1`.
- Model custom và VLM được chấm trên **cùng clip test của 5 fold**.
- Dấu `*` nghĩa là khác biệt có ý nghĩa thống kê so với custom v4 (kiểm định McNemar, p < 0,05).
- Số custom trong hai bảng dưới là bản train ngày 2026-10-21, thấp hơn bản tốt nhất ở mục 1.

### NSVA bóng rổ (8 329 clip test, macro-F1 trung bình 5 fold)

| # | Model | Loại | Tham số (train) | Macro-F1 | Accuracy | F1 made | F1 miss | F1 no-event |
|---|---|---|---|---|---|---|---|---|
| 1 | **custom v4** | train | 265M (23M) | **0,816** ± 0,008 | 0,817 | 0,868 | 0,806 | 0,774 |
| 2 | custom v3 | train | 197M (18M) | 0,815 ± 0,009 | 0,816 | 0,862 | 0,801 | 0,783 |
| 3 | custom v5 | train | 169M (17M) | 0,747 ± 0,010* | 0,747 | 0,766 | 0,737 | 0,737 |
| 4 | Qwen3.5-9B | VLM zero-shot | 9,41B | 0,621 ± 0,011* | 0,626 | 0,778 | 0,574 | 0,512 |
| 5 | Qwen3-VL-8B | VLM zero-shot | 8,77B | 0,442 ± 0,008* | 0,545 | 0,682 | 0,616 | 0,030 |
| 6 | InternVL3.5-8B | VLM zero-shot | 8,53B | 0,429 ± 0,011* | 0,534 | 0,638 | 0,618 | 0,030 |
| 7 | Qwen3-VL-4B | VLM zero-shot | 4,44B | 0,419 ± 0,008* | 0,522 | 0,621 | 0,607 | 0,030 |
| 8 | InternVL3.5-4B | VLM zero-shot | 4,73B | 0,365 ± 0,011* | 0,480 | 0,489 | 0,590 | 0,018 |
| 9 | MiniCPM-V-4 | VLM zero-shot | 4,06B | 0,353 ± 0,016* | 0,458 | 0,420 | 0,579 | 0,060 |
| 10 | Qwen3.5-4B | VLM zero-shot | 4,54B | 0,238 ± 0,014* | 0,361 | 0,508 | 0,001 | 0,205 |
| 11 | Gemma4-E4B | VLM zero-shot | 8,00B | 0,184 ± 0,008* | 0,310 | 0,088 | 0,002 | 0,462 |

Hầu hết VLM gần như không bao giờ trả lời `no-event` (F1 ≈ 0,03).

### Pickleball (4 685 clip ~1 s: 1 859 hit, 967 miss, 1 859 no-event; macro-F1 trung bình 5 fold)

| # | Model | Loại | Tham số (train) | Macro-F1 | Accuracy | F1 hit | F1 miss | F1 no-event |
|---|---|---|---|---|---|---|---|---|
| 1 | **custom v4** | train | 246M (4M) | **0,939** ± 0,045 | 0,943 | 0,960 | 0,936 | 0,932 |
| 2 | custom v3 | train | 182M (3M) | 0,928 ± 0,056* | 0,935 | 0,952 | 0,908 | 0,935 |
| 3 | custom v5 | train | 155M (3M) | 0,856 ± 0,111* | 0,862 | 0,888 | 0,835 | 0,861 |
| 4 | InternVL3.5-8B | VLM zero-shot | 8,53B | 0,774 ± 0,107* | 0,805 | 0,791 | 0,666 | 0,888 |
| 5 | Qwen3.5-9B | VLM zero-shot | 9,41B | 0,731 ± 0,079* | 0,743 | 0,669 | 0,740 | 0,800 |
| 6 | Qwen3.5-4B | VLM zero-shot | 4,54B | 0,661 ± 0,083* | 0,692 | 0,619 | 0,580 | 0,795 |
| 7 | Qwen3-VL-8B | VLM zero-shot | 8,77B | 0,604 ± 0,043* | 0,645 | 0,424 | 0,577 | 0,814 |
| 8 | InternVL3.5-4B | VLM zero-shot | 4,73B | 0,593 ± 0,050* | 0,638 | 0,538 | 0,346 | 0,901 |
| 9 | Qwen3-VL-4B | VLM zero-shot | 4,44B | 0,579 ± 0,055* | 0,626 | 0,351 | 0,567 | 0,822 |
| 10 | MiniCPM-V-4 | VLM zero-shot | 4,06B | 0,480 ± 0,044* | 0,555 | 0,090 | 0,483 | 0,870 |
| 11 | Gemma4-E4B | VLM zero-shot | 8,00B | 0,418 ± 0,029* | 0,567 | 0,500 | 0,034 | 0,730 |

Mọi VLM đều thấp hơn cả 3 model custom một cách có ý nghĩa. Model custom chỉ train 3–23M tham số (phần head), còn encoder được giữ cố định. Bóng đá và UCF-101 chưa có kết quả VLM.

---

## 3. Mô hình

Cả ba version dùng chung một ý tưởng: **encoder pretrained giữ cố định** (VideoMAE cho video, BEATs cho audio, CLIP cho text và ảnh). Feature được trích **một lần** vào cache, và chỉ phần **head** nhỏ được train. Code: `common/models/v3.py`, `v4.py`, `v5.py`.

```
video ─ VideoMAE-base (ViT-B, frozen) ─ 16 token thời gian × (mean + lưới 2×2) [+ CLIP ảnh] ─┐
audio ─ BEATs (frozen) ─ token audio ───────────────────────────────────────────── (v3, v4) ─┤ head train
text  ─ CLIP text (frozen) ─ mô tả lớp / prompt / lời bình ASR ───────────────────── (v4, v5) ─┘
```

| | **v3** `AVFeatureModel` | **v4** `AVTFeatureModel` | **v5** `VideoTextFeatureModel` |
|---|---|---|---|
| Đầu vào | video + audio | video + audio + text | video + text (không audio) |
| Chiếu | Linear → LayerNorm → GELU → Dropout cho từng modality, về `fusion_dim` | như v3, thêm nhánh text | như v3 (video, text) |
| Kết hợp | cross-attention: video (query) ← audio (key/value), residual + LayerNorm | 2 cross-attention video ← audio và video ← text, rồi **gated fusion**: `F = V + α·A_ctx + β·T_ctx` (α, β học riêng cho từng token) | **text-conditioned gate**: `F_t = LN(V_t + g_t · s · T)`, với `g_t = σ(MLP([V_t; T]))` |
| Thời gian | positional embedding học được + Transformer encoder (pre-norm) | như v3 | như v3 |
| Đầu ra | MLP 2 lớp cho từng token thời gian → logits lớp | **text-anchored head**: cosine(token, embedding CLIP của mô tả từng lớp) × scale + bias | text-anchored head |
| Mức clip | trung bình logits các token hợp lệ (NSVA, bóng đá, UCF); pickleball giữ nhãn theo từng token | | |

Text ở v4/v5 **không chứa nhãn**:
- Pickleball và UCF-101 dùng mô tả cố định của từng lớp (giống nhau cho mọi mẫu).
- NSVA dùng một prompt cố định.
- Bóng đá dùng lời bình luận do Whisper chép lại từ âm thanh của chính clip.

---

## 4. Cách train chung

1. **Trích feature một lần** (GPU) vào `<môn>/cache/`. Frame được lấy theo thời gian, vì fps giữa các video khác nhau.
2. **Train head** trên cache:
   - AdamW, warmup tuyến tính rồi cosine;
   - bf16 autocast, gradient clip 1,0;
   - early stopping theo tập **valid**;
   - **test không bao giờ được dùng để chọn** (cấu hình, epoch, ngưỡng).
3. **Chia dữ liệu theo nhóm**, không rò rỉ giữa train và test: pickleball chia theo người chơi, NSVA theo trận, bóng đá theo part có sẵn, UCF-101 theo group chính thức.
4. GPU dùng chung: các `run.sh` tự chờ cho tới khi đủ bộ nhớ, thử lại khi gặp OOM, và bỏ qua các run đã xong.

---

## 5. Chi tiết từng môn

### 5.1 Pickleball: `pickleball/`

- **Dữ liệu:** `/data/Cuong/pickleball/data/raw`, gồm 115 video `.MOV` kèm `.json` (sự kiện `hit` / `miss` theo thời gian), 17 người chơi.
- **Nhiệm vụ:** gán nhãn từng token thời gian (≈ 0,067 s) là `no_event` / `hit` / `miss`, sau đó ghép thành sự kiện. **Chỉ số chính: event macro-F1** (khớp sự kiện trong ±0,5 s). Tên file (Trung/Truot…) không bao giờ là đầu vào.
- **Feature:**
  - VideoMAE-base fine-tune K400, lấy frame ở 30 fps, cửa sổ 32 frame (1,07 s), bước 16 → 16 token; mỗi token gồm mean + lưới 2×2.
  - BEATs base chạy trên đoạn 10 s, rồi nội suy về đúng thời điểm của từng token.
  - Text (v4/v5): 4 câu mô tả CLIP ViT-B/32 cho mỗi lớp.
- **Đánh giá:** 5 fold chia theo người chơi; tập val (early stopping) cũng tách theo người chơi.
- **Cấu hình tốt nhất** (mặc định của `train.py`, giống nhau cho cả 3 version):

| Tham số | Giá trị | Tham số | Giá trị |
|---|---|---|---|
| epochs / patience | 60 / 10 | batch | 128 |
| lr / weight decay / warmup | 3e-4 / 0,05 / 3 epoch | loss | focal γ = 2, class weight `rsqrt` |
| fusion dim / heads / temporal layers | 256 / 8 / 2 | dropout / head | 0,2 / MLP 256, dropout 0,3 |
| augmentation | feature noise 0,1, audio drop 0,2 | nhãn | ±0,5 s quanh mỗi cú đánh |
| chọn checkpoint | val loss (`--select loss`) | spatial | mean + lưới 2×2 |

- **Kết quả:**

| | event macro-F1 | token macro-F1 | hit F1 | miss F1 |
|---|---|---|---|---|
| v3 | 0,931 | 0,869 | 0,964 | 0,897 |
| v4 | **0,951** | 0,879 | 0,976 | 0,926 |
| v5 | 0,838 | 0,786 | 0,853 | 0,823 |

- **Checkpoint:** `pickleball/outputs/runs/<version>/fold_<k>/best.pt` (5 fold). API chấm điểm (nhánh `score-video`) đóng gói 5 fold v4 bằng `scripts/export_checkpoint.py`.
- **Lưu ý dữ liệu:** mỗi video gần như chỉ toàn hit hoặc toàn miss; khoảng 87 nhãn có thể bị thiếu (`analysis/find_label_errors.py`).

```bash
cd baseline/pickleball
./run.sh                                   # trích feature (nếu chưa có) + train v3/v4/v5
python train.py --version v4               # train một version
python analysis/visualize_dataset.py       # biểu đồ dữ liệu -> outputs/analysis/
```

### 5.2 Bóng rổ NSVA: `basketball/`

- **Dữ liệu:** `/data/Cuong/pickleball/data/bendmark/basketball`; clip nằm trong 2 file zip; file mapping 1-1 `1_video_co_nhan_khop_1-1.csv`. Tổng 8 329 clip, 96 trận, 3 lớp `no_event` / `hit` (made) / `miss`.
- **Đánh giá:** 5-fold `StratifiedGroupKFold` theo trận (fold-seed 42); val = 1/8 số trận của phần train.
- **Cấu hình tốt nhất** (mặc định của `train.py`, giống nhau cho cả 3 version):
  - đầu vào: VideoMAE self-supervised, mean + lưới 2×2, thêm view nửa sau của clip ghép theo chiều feature (`--tail`), BEATs base, prompt text cố định;
  - tham số: fusion 384, weight decay 0,05, token-drop 0,1, audio-dropout 0,2, epochs 50, patience 10, batch 256, lr 3e-4, warmup 3, label smoothing 0,1, cross / temporal heads 16 / 12, 2 lớp temporal, dropout 0,1, head MLP 256 với dropout 0,3;
  - 3 seed (42, 1, 2).
- **Kết quả** (trung bình ± độ lệch chuẩn qua 3 seed; mỗi seed là trung bình 5 fold):

| | Macro-F1 | Accuracy | F1 no_event | F1 hit | F1 miss |
|---|---|---|---|---|---|
| v3 | **0,832** ± 0,004 | 0,833 | 0,792 | 0,888 | 0,817 |
| v4 | 0,828 ± 0,001 | 0,829 | 0,788 | 0,881 | 0,815 |
| v5 | 0,796 ± 0,001 | 0,796 | 0,769 | 0,835 | 0,784 |

- **Lỗi chính:** nhầm no_event ↔ miss, vì bóng và vành rổ chỉ chiếm vài pixel ở độ phân giải 224 px.
- **Checkpoint:** `basketball/outputs/runs/seed_<s>/<version>/fold_<k>/best.pt`.

```bash
cd baseline/basketball
./run.sh        # trích feature + train v3/v4/v5 (3 seed) + results/RESULTS.md
```

### 5.3 Bóng đá SoccerNet shots: `football/`

- **Dữ liệu:** `/data/Cuong/pickleball/data/bendmark/football`. part3–6 là train (6 094 clip), part1 là valid (1 931), part2 là test (2 137). Mỗi clip dài 15 s, cú sút ở giây thứ 5. Có 3 nhãn: `goal` (16%), `saved`, `off_target`.
- **Phân tích dữ liệu:** `data_report/DATA_REPORT.md`.
- **Đầu vào** (giống nhau cho cả 3 version và cả 2 bài): `--backbone k400 --spatial grid --window both --beats base --text asr --clip-image`. Cụ thể:
  - VideoMAE K400 (mean + lưới 2×2) trên 2 view: toàn clip, và đoạn 3–11 s lấy dày gấp đôi;
  - CLIP ViT-B/16 ảnh;
  - BEATs base;
  - lời bình luận do Whisper large-v3-turbo chép lại, mã hoá bằng CLIP text (v4/v5).
- **Siêu tham số tốt nhất:** đầy đủ trong `run.sh`. Chúng được chọn trên valid bằng random search, mỗi cấu hình chạy nhiều seed.

| Bài | | lr | wd | fusion | layers | dropout / head | token / audio / text drop | label smooth | class weight | batch | epochs | mixup | EMA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 nhãn | v3 | 5e-4 | 0,1 | 768 | 2 | 0,1 / 0,2 | 0,1 / 0,2 / – | 0 | balanced | 64 | 40 | 0,2 | 0,999 |
| 3 nhãn | v4 | 5e-4 | 0,01 | 512 | 2 | 0,3 / 0,5 | 0,2 / 0 / 0,4 | 0 | none | 64 | 60 | 0 | 0,999 |
| 3 nhãn | v5 | 1e-3 | 0,05 | 384 | 1 | 0,1 / 0,3 | 0,2 / – / 0,2 | 0,2 | sqrt | 128 | 60 | 0,2 | 0,999 |
| 2 nhãn | v3 | 1e-4 | 0,05 | 768 | 2 | 0,1 / 0,2 | 0,1 / 0 / – | 0 | none | 64 | 40 | 0,2 | – |
| 2 nhãn | v4 | 5e-4 | 0,1 | 512 | 2 | 0,3 / 0,5 | 0,1 / 0 / 0,4 | 0 | sqrt | 256 | 60 | 0,2 | – |
| 2 nhãn | v5 | 5e-4 | 0,1 | 384 | 3 | 0,1 / 0,2 | 0 / – / 0 | 0 | sqrt | 64 | 60 | 0,2 | – |

- **Kết quả trên test** (trung bình ± độ lệch chuẩn qua 5 seed):

| | 3 nhãn: macro-F1 | 3 nhãn: accuracy | F1 goal / saved / off_target | 2 nhãn: macro-F1 | 2 nhãn: accuracy | F1 goal / saved |
|---|---|---|---|---|---|---|
| v3 | **0,807** ± 0,002 | 0,779 | 0,932 / 0,713 / 0,778 | **0,970** ± 0,002 | 0,984 | 0,948 / 0,991 |
| v4 | 0,807 ± 0,007 | 0,775 | 0,953 / 0,685 / 0,782 | 0,968 ± 0,002 | 0,983 | 0,946 / 0,990 |
| v5 | 0,786 ± 0,005 | 0,753 | 0,923 / 0,684 / 0,752 | 0,954 ± 0,004 | 0,976 | 0,922 / 0,986 |

- **Nhận xét:** goal là lớp dễ. Cặp khó là saved ↔ off_target, nên khi gộp hai lớp này thành bài 2 nhãn, điểm tăng từ khoảng 0,81 lên khoảng 0,97.
- **Checkpoint:** `football/outputs/runs/seed_<s>/<version>/best.pt` (3 nhãn) và `outputs/runs_binary/` (2 nhãn).

```bash
cd baseline/football
./run.sh                 # 3 nhãn -> results/RESULTS.md
./run.sh --binary        # 2 nhãn -> results_binary/RESULTS.md
```

### 5.4 UCF-101: `ucf101/`

- **Dữ liệu:** `/data/Cuong/pickleball/data/ucf101`, gồm 13 320 clip, 101 lớp, 3 split chính thức. 51% số clip có audio.
- **v3 / v4 / v5 trên feature cố định:**
  - feature: VideoMAE K400 (mean + lưới 2×2), 6 view (train lấy ngẫu nhiên một view, test lấy trung bình cả 6), CLIP ViT-B/16 ảnh, BEATs fine-tune AudioSet;
  - text = 3 câu mô tả cho mỗi lớp;
  - val = 3 group giữ lại; sau đó refit trên toàn bộ train;
  - tham số: batch 128, epochs 60, patience 12, lr 3e-4, wd 0,05, warmup 5, label smoothing 0,1, fusion 768, head MLP 512, token-drop 0,1, audio-dropout 0,3.
- **v3 fine-tune end-to-end** (`finetune.py`):
  - VideoMAE-K400 + head v3, 16 frame lấy kiểu TSN, audio là BEATs cached;
  - 25 epoch, batch 4 × gradient accumulation 8, layer decay 0,75, head lr 3e-4;
  - test 6 view (2 vị trí thời gian × 3 crop).
- **Kết quả** (top-1 %, trung bình 3 split): v3 **98,1** ± 0,7 (fine-tune: 98,1 ± 0,6), v4 97,8 ± 0,6, v5 97,9 ± 0,5. Để so sánh, VideoMAE ViT-B fine-tune (self-supervised) trong bài báo đạt 96,1, I3D two-stream đạt 98,0.
- **Checkpoint:** `ucf101/outputs/runs/frozen/<version>/split<k>.pt` và `outputs/runs/ft_k400_av/split<k>/final.pt`.

```bash
cd baseline/ucf101
./run.sh       # feature -> v3/v4/v5 frozen -> v3 fine-tune -> results/RESULTS.md
```

---

## 6. Cấu trúc thư mục

```
baseline/
├── README.md
├── common/
│   ├── media.py              # đọc audio, nạp VideoMAE / BEATs, pooling token, seed, logits mức clip
│   ├── pretrained/           # BEATs (base, ft) + unilm, VideoMAE-base (git bỏ qua)
│   └── models/               # v3.py (AVFeatureModel), v4.py (AVTFeatureModel), v5.py (VideoTextFeatureModel)
├── pickleball/               # pickleball_common.py, train.py, run.sh, analysis/
├── basketball/               # nsva_common.py, extract_features.py, train.py, summarize.py, run.sh, results/
├── football/                 # fb_common.py, analyze_dataset.py, extract_features.py, transcribe.py, train.py,
│                             # summarize.py, run.sh, data_report/, results/, results_binary/
└── ucf101/                   # config.py, ucf101_common.py, analyze_ucf101.py, extract_features.py, train.py,
                              # finetune.py, summarize.py, run.sh, analysis/, results/
# mỗi môn còn có (git bỏ qua): cache/ = feature cache, outputs/ = checkpoint + kết quả, logs/
```

Ba thư mục, ba nhánh độc lập, không dùng chung code hay model:

```
/data/Cuong/pickleball/
├── pickleball_track/                  # nhánh training_model_custom: chỉ có baseline/
├── pickleball_track_score_video/      # nhánh score-video: API chấm điểm (weights, VideoMAE, unilm riêng)
└── pickleball_track_llm_benchmark/    # nhánh llm_benchmark: benchmark VLM zero-shot (benchmark_vlm/)
```

- Môi trường: conda `anything`. Nếu `transformers` báo lỗi import do thiếu `jmespath` (boto3 → botocore), thêm `PYTHONPATH=<baseline>/.pydeps`.

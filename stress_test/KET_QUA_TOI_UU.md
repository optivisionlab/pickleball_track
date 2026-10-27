# KẾT QUẢ TỐI ƯU — AI SERVICE VÀ BACKEND HAUI FITNESS SCORE

- Thời gian: 27–28/10/2026. Đo trên server `fithaui-kc` (48 nhân, 123 GB RAM, NVIDIA A30), test chạy ngay trên server (127.0.0.1) bằng đúng bộ kịch bản trong [BAO_CAO_KIEM_THU.md](BAO_CAO_KIEM_THU.md).
- Phương án đã duyệt: [PHUONG_AN_TOI_UU.md](PHUONG_AN_TOI_UU.md).
- GPU: các benchmark AI (mục 2) chạy khi GPU trống. Từ 16:00 tiến trình `unsloth` dùng lại GPU (9,9 GB, ~90–100%), nên các bài E2E và giờ cao điểm sau tối ưu có tranh chấp GPU, còn số "trước" đo lúc GPU trống hoặc chỉ bị tranh chấp một phần. Phần kết quả về video ngắn vì thế là ước lượng thận trọng.

---

## 1. Tóm tắt

| Chỉ số | Trước | Sau | Thay đổi |
|---|---|---|---|
| Thông lượng API sinh viên (đã có token) | ~100–108 req/s | **~416–444 req/s** | **×4,1** |
| API, 100 user dồn dập: TB / p95 | 831 / 2.603 ms | **193 / 478 ms** | −77% / −82% |
| Lỗi hệ thống ở 1.000 user | 87% | **0%** | |
| Lỗi hệ thống ở 5.000 user | 99,75% | **0%** (444 req/s) | |
| Spike 1.000 user trong 5 s | 78% lỗi | **0% lỗi** | |
| 50 lượt đăng nhập đồng thời | **Backend bị OOM-kill** | 0% lỗi, TB 1,0 s, 49 lượt/s | hết sập |
| Đăng nhập 1 user | 165 ms, 6 lượt/s | 57 ms, 17 lượt/s | ×2,9 |
| Nộp video đồng thời (10–20 SV) | 16/20 thành công (lỗi 500) | **20/20** | hết race condition |
| Giờ cao điểm: API TB / p95 | 633 ms / 2,15 s | **46 ms / 159 ms** | −93% |
| Giờ cao điểm: video ngắn E2E p95 | **105 s** | **6,4 s** | −94% |
| AI video ngắn (gọi thẳng, 12 đồng thời) | 98 video/phút, **33% bị 429** | **124 video/phút, 0% 429** | +26%, hết từ chối |
| AI hỗn hợp: video ngắn p95 | 107 s | **8,8 s** | −92% |
| AI video dài 60 s HDR | ~1,2 video/phút | ~1,2 video/phút | không đổi (giới hạn CPU giải mã) |
| Message Kafka job / kết quả | 710 / 453 byte | **239 / 190 byte** | −66% / −58% |

---

## 2. AI service (`pickleball_track`, nhánh `score-video`)

### 2.1 Thay đổi

| Thay đổi | Mục đích |
|---|---|
| `SlotScheduler`: cấp slot cho video **ngắn trước**, slot dành riêng cho video ≤ 20 s, video chờ quá 60 s được ưu tiên (chống bỏ đói) | Video ngắn không phải chờ sau video dài |
| **3 instance AI dùng chung GPU** + **nginx load balancer** ở cổng 8888 (`least_conn`, chuyển sang instance khác khi gặp 429/503) | Tăng thông lượng, loại bỏ 429, dễ scale (`AI_REPLICAS`) |
| Hàng đợi 8 → 12/instance (tổng 36), giới hạn 4 lượt tải MinIO đồng thời, `Retry-After` theo thời gian job trung bình | Ít từ chối, không quá tải mạng/disk |
| `/stats` thêm: video chờ ngắn/dài, slot dành riêng, thời gian job TB, bộ giải mã | Giám sát |
| Request chỉ bắt buộc `job_id`, `bucket`, `object_key` (vẫn nhận đủ trường cũ) | Request gọn |
| Bộ giải mã FFmpeg tùy chọn (`PICKLEBALL_DECODER=ffmpeg`, mặc định **tắt**) | Giảm CPU giải mã video HDR |

Test: 155 đạt (thêm 12 test mới), 1 lỗi có sẵn từ trước (`test_full_pipeline_matches_original`, cũng lỗi trên code gốc).

### 2.2 Chọn số instance × slot (GPU trống, cùng 3 tải chuẩn)

| Cấu hình | W1: 48 video ngắn, 12 đồng thời | W2: 4 video 60 s | W3 hỗn hợp: video ngắn TB / p95 / thông lượng | VRAM |
|---|---|---|---|---|
| Cũ: 1 × 2 slot, hàng đợi 8 | 97,6 video/phút, **16/48 bị 429** | 1,2/phút | 24,1 s / **106,9 s** / 9,7 video/phút | ~3,5 GB |
| A: 1 × 3 slot | 105,8/phút, 0 lỗi | 1,3/phút | 12,0 s / 13,6 s / 18,7 | ~3,5 GB |
| B: 2 × 2 slot | 111,4/phút, 0 lỗi | 1,1/phút | 6,9 s / 7,9 s / 33,2 | ~6,2 GB |
| **C: 3 × 2 slot (đang chạy)** | **123,5/phút, 0 lỗi** | 1,2/phút | **5,5 s / 8,8 s / 38,8** | ~10,4 GB |

Video dài không nhanh hơn ở mọi cấu hình: 83% thời gian là giải mã HEVC 10-bit HDR bằng OpenCV (~28 nhân CPU cho một video), phần model trên GPU chỉ 17%.

### 2.3 Kiểm chứng bộ giải mã FFmpeg (16 video có nhãn, 13 người, cả Trúng và Trượt)

| | OpenCV (đang dùng) | FFmpeg |
|---|---|---|
| Điểm chấm giống nhau | — | **16/16 video** |
| Số hit/miss giống hệt | — | 10/16 (6 video lệch vài sự kiện) |
| Khớp sự kiện với nhãn ±0,5 s: precision / recall / F1 | 0,732 / 0,460 / **0,565** | 0,720 / 0,498 / **0,589** |
| Video HDR iPhone (.MOV) | 25,8–53,5 s | **19,8–38,9 s (nhanh hơn 25–35%)** |
| Video 8-bit (.mp4) | 8,2–12,1 s | 10,9–15,7 s (chậm hơn) |

Khuyến nghị: dữ liệu huấn luyện và video thật đều là iPhone HDR, nên **có thể bật FFmpeg** (`PICKLEBALL_DECODER=ffmpeg`, rồi `docker compose up -d`). Tốt nhất là chọn tự động theo loại video (FFmpeg cho 10-bit, OpenCV cho 8-bit) — phần này chưa làm.

### 2.4 Phát hiện về chất lượng model (ngoài phạm vi tối ưu)

Trong cả 8 video "Trượt" của bộ kiểm chứng, model **không nhận ra cú trượt nào** (0 miss) và còn đếm một số cú là "trúng". Vì vậy điểm luôn là 10 hoặc 0 (vd. `Dong_Truot` có 19 cú trượt theo nhãn nhưng được 10 điểm). Cần xem lại ngưỡng `MISS_THRESHOLD = 0.8` hoặc checkpoint `best.pt`.

---

## 3. Backend (`haui_sict_fitness_score`, nhánh `improve/ai-grading-pipeline`)

### 3.1 Thay đổi

| Thay đổi | Mục đích |
|---|---|
| Argon2id tham số OWASP (19 MiB, t=2, p=1), băm trên thread pool riêng 4 luồng, tự băm lại mật khẩu cũ khi đăng nhập | Hết OOM; RAM worker ~185 MB thay vì 450–613 MB |
| Cấp `attemptNo` mới khi trùng (nộp đồng thời), trả 409 nếu vẫn trùng | Hết lỗi 500 khi nộp đồng thời |
| Job Kafka/outbox chỉ giữ 9 trường cần thiết; job cũ được thu gọn khi publish; request AI tối thiểu, **bỏ ký presigned URL**; kết quả không lặp trường; JSON gọn UTF-8 | Message nhỏ hơn 58–66%, bớt 1 lần gọi MinIO/job |
| Ngưỡng làn large 90 s → 20 s | Video ngắn không chờ sau video ~1 phút |
| `/api/tasks`, danh sách bài tập của GV: 1 aggregate thay vì 3 truy vấn/bài tập | Bỏ N+1 |
| Compose: 4 worker uvicorn, backend 4 CPU / 2 GB, MongoDB 4 CPU / 2 GB / `pids_limit` 4096, `nofile` 65536, pool MongoDB 50/client, worker AI 6 job (khớp 3×2 slot), Kafka giữ 3 ngày / 256 MB mỗi partition | Chạy hết sức trên server; hết lỗi "Too many open files" và "Cannot fork" |

### 3.2 Đăng nhập

| Đồng thời | Trước: TB / lượt/s / lỗi | Sau: TB / p95 / lượt/s / lỗi | RAM backend sau |
|---|---|---|---|
| 1 | 165 ms / 6,0 / 0% | 57 ms / 68 ms / 17,4 / 0% | 33% của 2 GB |
| 5 | 1.313 ms / 3,8 / 0% | 59 ms / 81 ms / 82,7 / 0% | 33% |
| 10 | 3.666 ms / 2,7 / 0% | 128 ms / 182 ms / 76,5 / 0% | 33% |
| 20 | 20.708 ms / 0,9 / 4,2% | 364 ms / 498 ms / 53,9 / 0% | 33% |
| 50 | **sập (OOM)** / 99,8% | 1.002 ms / 1.309 ms / 48,9 / 0% | 33% |
| 100 | — | 1.993 ms / 2.406 ms / 48,8 / 0% | 34% |

### 3.3 Luồng sinh viên (7 API, token sẵn, dồn dập)

| User | Trước: req/s / TB / p95 / lỗi hệ thống | Sau: req/s / TB / p95 / lỗi hệ thống |
|---|---|---|
| 10 | 108,6 / 78 / 268 ms / 0% | **415,7 / 20 / 44 ms / 0%** |
| 50 | 108,4 / 385 / 621 ms / 0% | **434,6 / 96 / 163 ms / 0%** |
| 100 | 100,2 / 831 / 2.603 ms / 0% | **432,0 / 193 / 478 ms / 0%** |
| 200 | 100,0 / 1.664 / 3.600 ms / 0% | **431,8 / 386 / 888 ms / 0%** |
| 500 | 101,4 / 4.106 / 6.677 ms / 0% | **428,8 / 969 / 2.428 ms / 0%** |
| 1.000 | ~83 thành công/s, **87% lỗi** | **425,2 / 1.936 / 6.600 ms / 0%** |
| Spike 1.000 / 5 s | 97 thành công/s, **78% lỗi** | **424,6 / 2.251 / 3.017 ms / 0%** |
| Stress 2.000 | 78 thành công/s, **99,2% lỗi** | **397,7 / 3.694 ms / 0%** |
| Stress 5.000 | 49 thành công/s, **99,75% lỗi** | **444,2 / 8.447 ms / 0%** |

Lỗi 400 ở `/video-results/{id}/status` (~5,7% tổng) là do dữ liệu test (5/10 tài khoản chưa có bài nộp), không tính là lỗi hệ thống — giống báo cáo trước. Container không bị restart lần nào; sau stress hệ thống healthy ngay (0 s).

### 3.4 Nộp video đầu-cuối

| Bài | Trước: thành công / TB E2E / p95 / video/phút | Sau |
|---|---|---|
| shot1 × 1 người | 5/5 / 2,7 s / 3,1 s / 22 | 5/5 / **2,1 s** / 2,1 s / 28 |
| shot1 × 5 | 10/10 / 4,1 s / 5,9 s / 63 | 10/10 / **3,2 s** / 4,3 s / 82 |
| shot1 × 10 | **16/20** / 5,7 s / 7,6 s / 75 | **20/20** / 4,9 s / 6,5 s / **102** |
| shot1 × 20 | **16/20** / 9,5 s / 15,6 s / 62 | **20/20** / 7,8 s / 12,2 s / 99 |
| shot2 × 3 | 3/3 / 118 s / 147 s / 1,18 | 3/3 / 117 s / 149 s / 1,17 |
| shot2 × 6 | 6/6 / 211 s / 313 s / 1,15 | 6/6 / 200 s / 301 s / 1,19 |

### 3.5 Giờ cao điểm (5 phút: 100 SV duyệt web + 4 luồng nộp shot1 + 2 shot2)

| | Trước | Sau |
|---|---|---|
| API: TB / p50 / p95 / p99 | 633 / 395 / 2.152 / 4.612 ms | **46 / 17 / 159 / 513 ms** |
| `/api/tasks` p95 | 5.125 ms | **196 ms** |
| CPU backend / MongoDB | 148% / 152% (chạm trần 150%) | 210% / 152% (trần 400%) |
| 40 bài shot1: TB / p95 / max E2E | 14,5 / **105,3** / 106,8 s | **5,0 / 6,4 / 6,9 s** |
| Thông lượng shot1 | 16,3 video/phút | **46,2 video/phút** |
| 2 bài shot2 | 108,8 s | 118,1 s (GPU đang bị tiến trình khác dùng 100%) |

### 3.6 Kích thước message

| Message | Trước | Sau |
|---|---|---|
| Job (`video-grading-jobs-*`) | 710 byte, 17 trường (task/course/sport/user id, `videoUrl`, `minioBucket` trùng, `detail`...) | **239 byte**, 9 trường |
| Kết quả (`video-grading-results`) | 453 byte (`score`/`aiScore`, `feedback`/`aiComment` lặp, tiếng Việt dạng `\uXXXX`) | **190 byte** |
| Request HTTP gọi AI | 15 trường + presigned URL ký mỗi lần | 7 trường, không ký URL |
| Lưu giữ Kafka | 7 ngày, không giới hạn dung lượng | 3 ngày, tối đa 256 MB/partition |

---

## 4. Triển khai hiện tại và cách quay lại

- AI: `cd /data/Cuong/pickleball/pickleball_track_score_video/score_video_block && docker compose up -d` (project `pickleball-ai`, 3 instance + nginx ở cổng 8888). Chỉnh bằng biến `AI_REPLICAS`, `PICKLEBALL_INFERENCE_SLOTS`, `PICKLEBALL_DECODER`.
- Quay lại AI cũ: `docker compose -p pickleball-ai down && docker update --restart=unless-stopped pickleball-score && docker start pickleball-score` (container cũ đang dừng, chưa xóa).
- Quay lại backend cũ: image cũ còn tag `haui-fitness-backend:pre-opt`, `haui-fitness-ai-worker:pre-opt`.
- **Thứ tự triển khai bắt buộc:** AI mới trước, backend mới sau (backend mới gửi request tối giản mà AI cũ không chấp nhận).

## 5. Commit (chỉ ở máy server, chưa push)

| Repo / nhánh | Commit |
|---|---|
| `pickleball_track` / `score-video` (worktree `/data/Cuong/pickleball/pickleball_track_score_video`) | `3f8ff05` mở rộng AI; `95c414b` mặc định 3 instance |
| `haui_sict_fitness_score` / `improve/ai-grading-pipeline` | `f2f62b4` tối ưu chính; `b1b4e1f` sửa RAM băm mật khẩu; `1d6d55d` MongoDB pids/pool |

Thay đổi chưa commit của bạn trong backend (bỏ giới hạn số lần nộp, `AI_MINIO_ENDPOINT`, seed, frontend) được giữ nguyên, không nằm trong các commit trên. Nhánh backend vẫn chậm hơn remote 1 commit (`e52c1a7`, sửa đúng các file frontend bạn đang sửa dở) — cần pull/rebase trước khi push.

## 6. Còn tồn tại

- Video dài vẫn ~1,2 video/phút (giới hạn CPU giải mã HDR). Hướng xử lý: bật FFmpeg (mục 2.3), giải mã bằng GPU (NVDEC), hoặc dùng model lấy mẫu 30 fps (code huấn luyện v3+ đã làm vậy).
- Model không phát hiện cú trượt (mục 2.4).
- Test có sẵn đang lỗi, không do lần sửa này: `test_guards.py` (Guard 4 mâu thuẫn với thay đổi chưa commit bỏ giới hạn nộp), các test tích hợp dùng `pipeline_harness` (cũng lỗi trên image cũ), `test_kafka_flow.py` (cần Kafka ở `localhost:9092`).
- GPU dùng chung với tiến trình huấn luyện `unsloth`.
- Không sửa FE (gzip/cache nginx) vì nằm ngoài phạm vi đã chọn.

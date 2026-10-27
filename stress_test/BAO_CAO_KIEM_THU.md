# BÁO CÁO KIỂM THỬ HIỆU NĂNG — HỆ THỐNG HAUI FITNESS SCORE

- Ngày thực hiện: 27/10/2026 (12:30 – 14:00)
- Hệ thống: FE `http://100.95.228.82:4200` · BE `http://100.95.228.82:8000` · AI `http://100.95.228.82:8888`
- Công cụ: Apache JMeter 5.6.3 (chế độ CLI), script Python (httpx async) cho luồng nộp video, giám sát bằng `docker stats` + `nvidia-smi` + AI `/stats` mỗi 2 giây
- Dữ liệu video: `shot1` (2 file, 3 giây, 1080×1920 30fps, 2,6 MB và 4,1 MB) · `shot2` (1 file, 60 giây, 1920×1080 60fps HEVC, 100,7 MB)
- Quy mô: **~5,12 triệu request HTTP** (JMeter, 41 bài test) + **114 lượt nộp video đầu-cuối** (106 thành công) + 134 job gọi thẳng AI

---

## 1. Tóm tắt kết quả

### 1.1 Năng lực hệ thống (đo trên server, không tính mạng)

| Thành phần | Thời gian phản hồi (1 request, không tải) | Thông lượng tối đa | Số request/user đồng thời chịu được | Điểm gãy |
|---|---|---|---|---|
| **FE** (nginx, trang + 11 file JS/CSS/ảnh) | 1,6 ms/file · 24 ms/trang | **~4.100 req/s** (~340 lượt tải trang/s) | **1.000 user, 0% lỗi** (trung bình 2,6 s/trang) | Không gãy; nghẽn ở giới hạn 0,5 CPU |
| **BE** endpoint `/` (không logic) | 5,6 ms | ~1.700 req/s | 500 user, 0% lỗi | — |
| **BE API sinh viên** (7 API có xác thực + MongoDB) | 40–230 ms (10 user) | **~100–108 req/s** | **~50 user liên tục** (p95 0,62 s); 500 user vẫn 0% lỗi nhưng chậm 4,1 s | **1.000 user → 88% lỗi** (hết file descriptor) |
| **BE đăng nhập** | 165 ms | **6 lần/s** | 5–10 lượt đồng thời | **50 lượt đồng thời → backend bị OOM, khởi động lại** |
| **AI** – video 3 s (shot1) | **1,04 s** | **~90 video/phút** (1,5 video/s) | 8 job (2 chạy + 6 chờ) | Job thứ 9 trở đi bị từ chối 429 |
| **AI** – video 60 s (shot2) | **47,9 s** | **~1,2 video/phút** (~72 video/giờ) | 8 job | Job thứ 9 trở đi bị từ chối 429 |
| **E2E nộp bài** shot1 (qua BE+Kafka+AI) | 2,1–2,7 s | **~62–75 video/phút** | 20 người nộp cùng lúc: tổng 9,5 s/bài | 20% lỗi 500 khi 1 SV nộp trùng nhiều lượt (race condition) |
| **E2E nộp bài** shot2 | 53,5 s | **~1,15 video/phút** | 6 người cùng lúc: người cuối chờ 5,2 phút | Không lỗi |

### 1.2 Trải nghiệm từ máy người dùng (qua mạng Tailscale, RTT ~94 ms, upload ~2,1 MB/s)

| Thao tác | Kết quả |
|---|---|
| Mở trang FE (50 user cùng lúc) | 10,2 s/trang (bị giới hạn bởi băng thông; trang nặng 551 KB do **không bật gzip**) |
| API sinh viên (50 user cùng lúc) | trung bình 459 ms, p95 1,48 s, 0% lỗi hệ thống |
| Nộp video shot1 (4 MB) | upload 1,7 s + AI chấm 2,1 s = **3,9 s** |
| Nộp video shot2 (100 MB) | upload 44,6 s + AI chấm 62,5 s = **107 s** |

### 1.3 Lỗi và điểm nghẽn phát hiện (xếp theo mức độ)

| # | Mức | Vấn đề | Bằng chứng |
|---|---|---|---|
| 1 | **Nghiêm trọng** | Đăng nhập dùng Argon2 mặc định (64 MB RAM/lần băm). ~40 lượt song song × 64 MB > giới hạn 768 MB → **backend bị OOM-kill**. Thông lượng login còn **giảm** khi tăng tải (6 → 0,9 req/s) | `be_login_u50`: sự kiện `oom` + `die` + `start` của `haui-fitness-backend-1`; RAM 172 MB (1 user) → 768 MB (20 user) |
| 2 | **Cao** | Container backend chỉ được mở **1.024 file** (soft limit mặc định, compose không khai `ulimits`). ≥1.000 kết nối → `OSError: [Errno 24] Too many open files` → không mở được kết nối MongoDB → HTTP 500 + rớt kết nối | 2.597 lỗi `ServerSelectionTimeoutError ... Too many open files` trong log; `/proc/1/limits` = 1024 |
| 3 | **Cao** | **Race condition khi nộp bài**: cùng 1 SV nộp đồng thời nhiều lượt cho 1 bài tập → tính trùng `attemptNo` → `E11000 duplicate key` → HTTP 500 | `e2e_shot1_c10`, `c20`: 4/20 lượt lỗi `confirm-upload 500`; log `DuplicateKeyError ... taskId_1_userId_1_attemptNo_1` |
| 4 | **Cao** | **Video ngắn bị kẹt sau video dài**: ngưỡng phân làn 90 s nên video 60 s vẫn vào làn "nhỏ"; AI chỉ có 2 slot dùng chung → video 3 s phải chờ video 60 s | Giờ cao điểm: shot1 trung bình 14,5 s nhưng **p95 = 105 s**, max 107 s (bình thường 2–3 s) |
| 5 | Trung bình | Thông lượng API có xác thực chỉ ~100 req/s: 1 tiến trình uvicorn (1,5 CPU), middleware đọc → parse → serialize lại toàn bộ JSON mọi response, mỗi request truy vấn user từ MongoDB; MongoDB cũng chạm trần 1,5 CPU khi tải cao | CPU backend 148% / mongo 152% (giới hạn 150%) ở giờ cao điểm; `/` không qua middleware đạt 1.700 req/s |
| 6 | Trung bình | `/api/tasks` chậm gấp 3–5 lần các API khác (231 ms so với 40–66 ms ở 10 user) | `be_flow_u10` |
| 7 | Trung bình | AI bị giới hạn bởi **CPU giải mã video**, không phải GPU: 2 video 60 s song song mất 93 s/video (không nhanh hơn chạy lần lượt); AI dùng tới 37 nhân CPU, GPU chỉ 6–65% | `ai_shot2_c2`: 1,28 video/phút ≈ `ai_shot2_c1`: 1,25 video/phút |
| 8 | Thấp | FE không bật gzip và không có `Cache-Control` cho file đã hash → 551 KB/lượt (gzip chỉ còn 149 KB, giảm 73%) | `curl -H 'Accept-Encoding: gzip'` không trả `Content-Encoding` |
| 9 | Thấp | Khi CPU bị chiếm bởi băm mật khẩu, Kafka consumer của backend mất heartbeat | Log `Heartbeat session expired - marking coordinator dead` |

**Điểm tốt:** cơ chế chống quá tải của AI hoạt động đúng (từ chối 429 trong 30–54 ms, không treo, không tràn GPU); backend tự khởi động lại sau OOM và **hồi phục ngay (0 s)** sau khi hết tải stress; không mất bài nộp nào (mọi lượt đã `confirm` đều được chấm); FE không lỗi ở 1.000 user.

---

## 2. Môi trường kiểm thử

### 2.1 Kiến trúc

```
Trình duyệt ──► FE nginx :4200 (0,5 CPU, 128 MB)
             └► BE FastAPI :8000 (APP_ROLE=api, 1 uvicorn, 1,5 CPU, 768 MB)
                   │  presigned PUT ──► MinIO :9000
                   ├► MongoDB (1,5 CPU, 768 MB) · Redis
                   └► Kafka (video-grading-jobs-small / -large)
                          └► AI worker (APP_ROLE=worker, 1 CPU, 512 MB, làn nhỏ 2 / làn lớn 1)
                                 └► AI pickleball-score :8888 (VideoMAE + BEATs, 2 slot suy luận, hàng đợi tối đa 8)
```

### 2.2 Phần cứng

| | Server `fithaui-kc` | Máy client |
|---|---|---|
| CPU | 48 nhân | 12 nhân |
| RAM | 123 GB | 7,4 GB |
| GPU | NVIDIA A30 24 GB | — |
| Mạng tới server | — | Tailscale, RTT 49–158 ms (tb 94 ms), upload ~2,1 MB/s |

### 2.3 Lưu ý ảnh hưởng tới số liệu

- **GPU dùng chung:** trong lúc test, một tiến trình khác (môi trường `unsloth`, ~11–21 GB VRAM) chạy bật/tắt trên cùng GPU A30. Các bài AI shot1 và `ai_shot2_c1/c2` chạy khi GPU gần trống; `ai_shot2_c4`, `ai_shot2_c10`, E2E shot2 và giờ cao điểm có tranh chấp GPU. Số liệu AI ở các bài sau có thể chậm hơn khi GPU riêng.
- Test tải chạy **ngay trên server** (`127.0.0.1`) để đo năng lực thuần, loại bỏ nút cổ chai mạng. Test từ client đo riêng ở mục 3.9.
- Lỗi 400 ở `GET /api/video-results/{id}/status` (~40–50%) **là do dữ liệu test**: 5/10 tài khoản (lớp Điền kinh) chưa có bài nộp nên không có ID để hỏi trạng thái. Không phải lỗi hệ thống; đã loại khỏi kết luận.
- Bài test luồng sinh viên dùng **token đăng nhập sẵn** (10 tài khoản) để tách biệt khỏi lỗi đăng nhập (mục 1.3 #1).

---

## 3. Kịch bản và kết quả chi tiết

Quy ước: *closed-loop* = mỗi user ảo gửi request kế tiếp ngay khi nhận phản hồi (think time 0), tức là tải nặng hơn nhiều so với người thật. Thời gian tính bằng ms trừ khi ghi khác. `p95` = 95% request nhanh hơn giá trị này.

### 3.1 TC-FE-01: Tải trang Frontend

- **Mục tiêu:** đo khả năng phục vụ trang Angular (index.html + 6 chunk JS + main.js + CSS + logo, 12 request/lượt).
- **Kịch bản:** `jmeter/fe_page.jmx`. Mỗi user: GET `/` → trích danh sách asset từ HTML → tải lần lượt từng asset (gộp thành giao dịch *PAGE LOAD*). Closed-loop, ramp-up 10 s, giữ 60 s.
- **Tiêu chí đạt:** lỗi < 1%, p95 trang < 3 s.

| User đồng thời | Tổng request | Lỗi | TB/request | p95/request | TB/trang | p95/trang | Thông lượng (req/s) | Lượt trang/s | CPU nginx |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 242.410 | 0% | 2,0 | 1 | 24 | 80 | 4.044 | 337 | 54% |
| 50 | 228.686 | 0% | 11,8 | 92 | 141 | 202 | 3.810 | 317 | 54% |
| 100 | 209.277 | 0% | 26,0 | 98 | 312 | 499 | 3.490 | 290 | 54% |
| 200 | 221.755 | 0% | 49,2 | 193 | 589 | 993 | 3.692 | 306 | 54% |
| 500 | 248.602 | 0% | 109,9 | 300 | 1.315 | 2.089 | 4.129 | 340 | 54% |
| 1.000 | 252.425 | 0% | 217,8 | 606 | 2.601 | 4.387 | 4.172 | 340 | 54% |

**Nhận xét:** đạt tới 500 user; 1.000 user vẫn 0% lỗi nhưng p95 trang 4,4 s. Thông lượng đứng ở ~4.000 req/s vì nginx luôn ở 54% = giới hạn `cpus: 0.50`; độ trễ nhảy theo bậc 100 ms (dấu hiệu CPU throttling của Docker, chu kỳ 100 ms). Tăng `cpus` cho FE hoặc bật gzip/cache sẽ nâng trần.

### 3.2 TC-BE-01: Endpoint gốc của Backend (baseline)

- **Kịch bản:** `jmeter/be_simple.jmx -Jpath=/`, closed-loop, giữ 60 s. Đo chi phí tối thiểu của FastAPI/uvicorn (không middleware, không DB).

| User | Request | Lỗi | TB | p50 | p95 | p99 | max | req/s |
|---|---|---|---|---|---|---|---|---|
| 10 | 102.201 | 0% | 5,6 | 5 | 7 | 11 | 91 | 1.705 |
| 50 | 97.566 | 0% | 29,4 | 27 | 73 | 79 | 94 | 1.628 |
| 100 | 89.700 | 0% | 64,1 | 58 | 110 | 117 | 213 | 1.496 |
| 200 | 84.997 | 0% | 135,4 | 123 | 185 | 193 | 267 | 1.416 |
| 500 | 76.251 | 0% | 378,1 | 406 | 431 | 459 | 812 | 1.268 |

### 3.3 TC-BE-02: Đăng nhập

- **Kịch bản:** `jmeter/be_login.jmx`, 10 tài khoản xoay vòng, POST `/api/auth/login`, giữ 60 s.
- **Tiêu chí đạt:** p95 < 2 s, không lỗi.

| Lượt đăng nhập đồng thời | Request | Lỗi | TB | p95 | p99 | Thông lượng (login/s) | RAM backend đỉnh | Kết quả |
|---|---|---|---|---|---|---|---|---|
| 1 | 359 | 0% | 165 | 193 | 214 | **6,01** | 172 MB | Đạt |
| 5 | 226 | 0% | 1.313 | 1.603 | 1.784 | 3,75 | 430 MB | Đạt |
| 10 | 163 | 0% | 3.666 | 4.902 | 5.293 | 2,67 | 692 MB (90%) | Không đạt |
| 20 | 71 | 4,2% | 20.708 | 39.945 | 60.057 | 0,93 | 768 MB (100%) | Không đạt (timeout) |
| 50 | 59.732 | **99,8%** | — | — | — | — | 768 MB | **Sập: OOM-kill, container khởi động lại** |

**Nhận xét:** đây là điểm yếu nặng nhất. Khi cả lớp đăng nhập cùng lúc đầu giờ thi (vd. 40 SV bấm đăng nhập trong 5 giây), backend có thể bị kill và toàn bộ người dùng khác cũng mất kết nối.

### 3.4 TC-BE-03: Luồng sinh viên (API có xác thực)

- **Kịch bản:** `jmeter/be_user_flow_tok.jmx`. Mỗi vòng 7 API: `/auth/me`, `/enrollments/my-courses`, `/tasks`, `/video-results`, `/video-results/{id}/status`, `/sports`, `/notifications`. Closed-loop, ramp-up 20 s, giữ 60 s.
- **Tiêu chí đạt:** lỗi hệ thống < 1%, p95 < 1 s.

| User | Request | Lỗi hệ thống* | TB | p50 | p90 | p95 | p99 | req/s | CPU BE / Mongo |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 6.508 | 0% | 78 | 58 | 206 | 268 | 332 | 108,6 | 110% / 104% |
| 50 | 6.520 | 0% | 385 | 390 | 562 | 621 | 1.567 | 108,4 | 127% / 81% |
| 100 | 6.059 | 0% | 831 | 724 | 2.299 | 2.603 | 2.843 | 100,2 | 112% / 109% |
| 200 | 6.097 | 0% | 1.664 | 1.713 | 3.045 | 3.600 | 3.966 | 100,0 | 126% / 114% |
| 500 | 6.368 | 0% | 4.106 | 4.423 | 6.351 | 6.677 | 7.696 | 101,4 | 132% / 99% |
| 1.000 | 43.682 | **87%** | 1.209 | 5 | 5.280 | 9.430 | 16.110 | 674,8 (thành công ~83) | 121% / 102% |

\* Không tính lỗi 400 ở `/status` do dữ liệu test (mục 2.3).

Thời gian trung bình từng API ở 10 user: `/auth/me` 41 · `/notifications` 50 · `/sports` 53 · `/enrollments/my-courses` 65 · `/video-results` 66 · `/status` 41 · **`/tasks` 231**.

**Nhận xét:** thông lượng phẳng ở ~100 req/s từ 10 user trở lên — thêm user chỉ làm hàng đợi dài ra (đúng định luật Little: 500 user / 101 req/s ≈ 4,9 s). Đạt tiêu chí tới **~50 user closed-loop**. Với người dùng thật (mỗi người ~1 thao tác/5 s), 100 req/s tương đương khoảng **400–500 người dùng đang hoạt động**; nên vận hành ở ≤70% (≈300–350 người).

### 3.5 TC-BE-04: Stress (đến khi hỏng) và Spike

- **Stress:** luồng sinh viên, 2.000 / 3.000 / 5.000 user, ramp-up 30 s, giữ 60 s.
- **Spike:** 1.000 user ập vào trong 5 s.

| Bài | Request | Lỗi | Request thành công/s | Container sập? | Hồi phục sau test |
|---|---|---|---|---|---|
| Spike 1.000 user / 5 s | 29.255 | 78,4% | 97,3 | Không | ngay |
| Stress 2.000 user | 737.340 | 99,2% | 77,9 | Không | ngay |
| Stress 3.000 user | 1.016.745 | 99,4% | 69,9 | Không | ngay |
| Stress 5.000 user | 1.332.112 | 99,8% | 49,1 | Không | 0 s (healthy ngay) |

**Nhận xét:** API đọc không làm sập backend (RAM chỉ 24%); hệ thống từ chối kết nối khi vượt 1.024 file descriptor, phần thành công giảm dần theo tải. Điểm sập thật duy nhất tìm thấy là bão đăng nhập (3.3).

### 3.6 TC-AI-01/02: Gọi thẳng AI service (cổng 8888)

- **Kịch bản:** `jmeter/ai_direct.jmx`, POST `/api/v1/process` với video đã có trên MinIO, mỗi request một `job_id` mới (UUID) để **không dính cache kết quả** của AI. Tất cả user gửi cùng lúc (ramp 0), mỗi user lặp N lần.

**shot1 — video 3 s (4,1 MB, H.264)**

| Đồng thời | Gửi | Thành công | Bị từ chối 429 | TB (thành công) | min | max | Thông lượng | GPU đỉnh | CPU AI đỉnh |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 10 | 0 | **1,04 s** | 1,00 | 1,14 | 57 video/phút | 0%* | 9,8 nhân |
| 2 | 16 | 16 | 0 | 1,27 s | 1,09 | 1,81 | **92 video/phút** | 48% | 12,6 nhân |
| 4 | 20 | 20 | 0 | 3,08 s | 1,82 | 3,50 | 74 video/phút | 58% | 17,5 nhân |
| 8 | 32 | 32 | 0 | 5,05 s | 1,63 | 6,23 | 87 video/phút | 88% | 16,6 nhân |
| 12 | 36 | 24 | 12 (33%) | 4,55 s | 1,47 | 5,64 | 92 video/phút | — | 13,4 nhân |

\* Lấy mẫu 2 s nên không bắt được đợt GPU ngắn ~1 s.

**shot2 — video 60 s (100,7 MB, HEVC 1080p60)**

| Đồng thời | Gửi | Thành công | 429 | TB (thành công) | min | max | Thông lượng | GPU tb / VRAM đỉnh | CPU AI đỉnh |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 0 | **47,9 s** | 47,4 | 48,5 | 1,25 video/phút | 6% / 3,5 GB | 28 nhân |
| 2 | 4 | 4 | 0 | 93,5 s | 89,2 | 98,3 | 1,28 video/phút | 9% / 3,5 GB | 37 nhân |
| 4 | 4 | 4 | 0 | 160,1 s | 107,7 | 210,8 | 1,14 video/phút | 38% / 13,3 GB† | 37 nhân |
| 10 | 10 | 8 | 2 (20%) | 265,1 s | 105,0 | **422,6 s** | 1,14 video/phút | 65% / 14,9 GB† | 36 nhân |

† Có tiến trình khác dùng chung GPU.

**Nhận xét:**
- Thời gian xử lý ≈ **0,8 × thời lượng video** (60 s video → 48 s). Video ngắn có chi phí cố định ~1 s.
- Chạy 2 slot song song **không tăng thông lượng** với video dài: nút thắt là giải mã HEVC 1080p60 + tiền xử lý trên CPU (28–37 nhân), GPU gần như rảnh. Cấu hình `gpu_preprocess`, giảm fps/độ phân giải trước khi giải mã, hoặc giải mã bằng NVDEC sẽ có lợi nhất.
- Cơ chế bảo vệ đúng thiết kế: tối đa 8 job, job thứ 9+ bị trả 429 sau 30–54 ms, backend sẽ chờ `Retry-After` rồi gửi lại.

### 3.7 TC-E2E-01: Nộp video đầu-cuối qua Backend

- **Kịch bản:** `scripts/e2e_video.py`. Mỗi lượt: POST `/video-results/upload-url` → PUT video thẳng lên MinIO → POST `/confirm-upload` → hỏi `/status` mỗi 1 s tới khi AI chấm xong. 6 tài khoản SV lớp Pickleball, bài TX1 (shot1) và TX2 (shot2). Chạy trên server.
- **Đo:** thời gian nộp (3 bước đầu), thời gian chờ chấm (Kafka → worker → AI → kết quả), tổng E2E.

| Bài | Người nộp cùng lúc | Thành công | Thời gian chạy | Thông lượng | Nộp TB / p95 | Chờ chấm TB / max | **E2E TB / p95 / max** |
|---|---|---|---|---|---|---|---|
| shot1 baseline | 1 | 5/5 | 13,5 s | 22 video/phút | 74 / 80 ms | 2,6 / 3,0 s | **2,7 / 3,1 / 3,1 s** |
| shot1 | 5 | 10/10 | 9,6 s | 63 video/phút | 165 / 232 ms | 3,9 / 6,2 s | 4,1 / 5,9 / 6,4 s |
| shot1 | 10 | 16/20 | 12,8 s | **75 video/phút** | 244 / 472 ms | 5,5 / 8,1 s | 5,7 / 7,6 / 8,2 s |
| shot1 | 20 | 16/20 | 15,6 s | 62 video/phút | 985 / 1.172 ms | 8,5 / 14,5 s | 9,5 / 15,6 / 15,6 s |
| shot2 baseline | 1 | 1/1 | 53,5 s | 1,1 video/phút | 477 ms | 53,0 s | **53,5 s** |
| shot2 | 3 | 3/3 | 151,9 s | 1,18 video/phút | 2,3 s | 116 / 150 s | 118 / 147 / 152 s |
| shot2 | 6 | 6/6 | 313,0 s | 1,15 video/phút | 5,0 s | 206 / 308 s | 211 / 313 / **313 s** |

Lỗi ở bài 10 và 20 người: `confirm-upload` trả 500 do `E11000 duplicate key` (mục 1.3 #3) — script dùng 6 tài khoản nên mỗi SV có 2–4 lượt nộp chồng nhau.

### 3.8 TC-MIX-01: Giờ cao điểm (hỗn hợp, 5 phút)

- **Kịch bản:** 100 SV duyệt web (luồng 3.4, nghỉ 1 s giữa các vòng) trong 300 s; từ giây 30 có 4 SV liên tục nộp shot1 (tổng 40 bài) và 2 SV nộp shot2.

| Thành phần | Kết quả |
|---|---|
| 100 SV duyệt web | 17.739 request, 0% lỗi hệ thống, TB 633 ms, p50 395 ms, p95 2,15 s, p99 4,6 s, 59 req/s; `/tasks` p95 5,1 s |
| Tài nguyên | CPU backend 148%, MongoDB **152%** (cả hai chạm giới hạn 150%), GPU 100% |
| 40 bài shot1 | 40/40 thành công, E2E TB **14,5 s**, **p95 105 s**, max 107 s (bình thường 2–3 s) |
| 2 bài shot2 | 2/2 thành công, E2E 108,8 s |

**Nhận xét:** video ngắn bị chặn sau video dài (head-of-line blocking): video 60 s < ngưỡng làn 90 s nên đi chung làn "small", 2 slot AI bị chiếm ~100 s. Hạ `AI_LANE_THRESHOLD_SEC` (vd. 15–20 s) và **dành riêng 1 slot AI cho video ngắn** sẽ khắc phục.

### 3.9 TC-CLIENT-01: Đo từ máy người dùng (qua mạng thật)

| Bài | User | Kết quả |
|---|---|---|
| FE tải trang | 50 | 3.292 request, 0% lỗi, TB 842 ms/file, **TB 10,2 s/trang**, p95 15,9 s — bị giới hạn bởi băng thông ~2,5 MB/s (551 KB/trang, không gzip) |
| API sinh viên | 50 | 6.019 request, 0% lỗi hệ thống, TB 459 ms, p95 1,48 s, 99,8 req/s (vẫn là trần của backend) |
| Nộp shot1 (4 MB) | 1 × 3 lượt | upload MinIO 1,68 s · nộp 1,84 s · chấm 2,06 s · **E2E 3,9 s** |
| Nộp shot2 (100 MB) | 1 | upload MinIO **44,6 s** · nộp 44,8 s · chấm 62,5 s · **E2E 107,3 s** |

---

## 4. Đề xuất khắc phục (theo thứ tự ưu tiên)

1. **Đăng nhập (sửa ngay):** giảm tham số Argon2 theo khuyến nghị OWASP cho server (vd. `PasswordHasher(memory_cost=19456, time_cost=2, parallelism=1)` ≈ 19 MB) **và** giới hạn số lượt băm đồng thời bằng semaphore (vd. 4–8); hoặc tăng `mem_limit` backend. Thêm rate-limit cho `/auth/login`. Tài khoản cũ vẫn xác minh được vì tham số nằm trong chuỗi hash; dùng `ph.check_needs_rehash()` để nâng cấp dần.
2. **File descriptor:** thêm `ulimits: { nofile: { soft: 65536, hard: 65536 } }` cho `backend` (và `ai-worker`) trong `docker-compose.yml`; đặt `maxPoolSize` hợp lý cho MongoClient.
3. **Race condition nộp bài:** bắt `DuplicateKeyError` trong `confirm_upload` và thử lại với `attemptNo` kế tiếp (hoặc cấp `attemptNo` bằng `findOneAndUpdate` + `$inc` trên một bộ đếm); trả 409 thay vì 500 nếu vẫn trùng. Phía FE nên khóa nút "Nộp" khi đang gửi.
4. **Phân làn AI:** hạ `AI_LANE_THRESHOLD_SEC` xuống ~15–20 s và cho AI service một slot riêng cho video ngắn (hoặc chạy 2 instance AI: 1 cho video ngắn, 1 cho video dài).
5. **Tăng thông lượng API:** chạy nhiều worker (`uvicorn --workers 3` hoặc gunicorn + uvicorn worker) với CPU tương ứng; thay `BaseHTTPMiddleware` bọc JSON bằng response model/`JSONResponse` bọc sẵn trong endpoint; cache user theo token (Redis, TTL ngắn); tối ưu `/api/tasks` (kiểm tra index/N+1); tăng CPU cho MongoDB.
6. **AI:** bật `PICKLEBALL_GPU_PREPROCESS`, giải mã bằng GPU (NVDEC) hoặc lấy mẫu khung hình trước khi resize; tách GPU khỏi tiến trình huấn luyện khác khi chạy thật.
7. **FE nginx:** bật `gzip on; gzip_types application/javascript text/css application/json;` và `Cache-Control: public, max-age=31536000, immutable` cho file đã hash; tăng `cpus` nếu cần >1.000 user.
8. **Upload video dài:** khuyến nghị nén/giảm độ phân giải phía client trước khi upload (100 MB mất ~45 s ở 17 Mbps), hoặc dùng multipart upload có hiển thị tiến độ.

---

## 5. So sánh với các hệ thống tương tự

### 5.1 Hệ thống phân tích video thể thao bằng AI

| Hệ thống | Cách xử lý | Thời gian trả kết quả (công bố) | So với HAUI Fitness |
|---|---|---|---|
| **HAUI Fitness Score** (đo được) | Server, VideoMAE + BEATs, A30 | Video 3 s: **2–4 s**; video 60 s: **~48–63 s** xử lý (~0,8× thời lượng) | — |
| **PB Vision** (pickleball) | Cloud, upload video trận | "Hầu hết video xong trong khoảng **30 phút**"; video bị gắn cờ có thể tới 24 giờ; người dùng báo 3–24 giờ ([help.pb.vision](https://help.pb.vision/articles/1981767-video-upload-troubleshooting), [Google Play](https://play.google.com/store/apps/details?id=com.pb.vision.app&hl=en_US)) | Nhanh hơn rất nhiều, nhưng PB Vision phân tích cả trận (tới 30 phút, 5 GB) với nhiều chỉ số hơn — không cùng bài toán |
| **Hudl Assist** (bóng đá, bóng rổ…) | Cloud + người phân tích | Thường **< 24 giờ**; khúc côn cầu có gói 12 giờ ([hudl.com](https://Hudl.com/products/assist/basketball), [hudl.com/blog](https://www.hudl.com/blog/were-breaking-the-ice-hudl-assist-is-coming-to-hockey)) | Hudl có người phân tích tham gia; HAUI hoàn toàn tự động |
| **SwingVision** (tennis, pickleball) | Chạy trên thiết bị (iPhone/iPad), thời gian thực | Nguồn tiếp thị nói xử lý **real-time**; báo cáo ITF mô tả dữ liệu đẩy lên server sau trận để xem lại ([ITF PAT report](https://www.itftennis.com/media/15294/pat-25-037-approval-report-swingvision.pdf), [App Store](https://apps.apple.com/MX/app/id989461317)) | SwingVision dùng mô hình nhẹ trên máy; HAUI dùng mô hình video+âm thanh nặng hơn trên server |

**Thông lượng mô hình:** VideoMAE-B (16 khung, 224×224, ~180 GFLOPs/clip) đạt **13,24 video/s** trên RTX 3090 FP16 khi chỉ đo suy luận ([MaMe & MaRe, arXiv 2604.13432](https://arxiv.org/pdf/2604.13432); GFLOPs theo [slide NeurIPS 2022](https://nips.cc/media/neurips-2022/Slides/65141.pdf)). HAUI đạt ~1,5 video ngắn/s **tính cả tải video, giải mã, tiền xử lý, BEATs và hậu xử lý**; với video dài, thời gian bị chi phối bởi giải mã CPU (GPU chỉ 6–9%) — tức là phần mô hình chưa phải nút thắt, còn nhiều dư địa tối ưu ở khâu giải mã.

### 5.2 Hệ thống LMS / web backend

| Hệ thống | Số liệu công bố | So với HAUI Fitness |
|---|---|---|
| **Moodle** (bài test chính thức 2.3 → 2.4) | JMeter 200 user đồng thời, kịch bản đăng nhập + xem khóa học + diễn đàn ([docs.moodle.org](https://docs.moodle.org:443/dev/Moodle_2.4_release_notes/Performance_tests)); cộng đồng báo đăng nhập 8–10 s với 200 user trên 2 server × 4 vCPU ([moodle.org](https://moodle.org/mod/forum/discuss.php?d=451810)) | HAUI đăng nhập 1 user 165 ms nhưng không chịu nổi 50 lượt đồng thời; API đọc 200 user closed-loop p95 3,6 s, 0% lỗi trên chỉ 1,5 CPU |
| **Canvas LMS** (tự host, cộng đồng) | Khoảng **25 user đồng thời / CPU** khi mỗi user thao tác 3–6 s/lần ([Canvas users group](https://groups.google.com/g/canvas-lms-users/c/MOEMyY3-VcE)) | HAUI: ~100 req/s trên 1,5 CPU ≈ 300–500 user hoạt động (1 thao tác/5 s) → **tương đương hoặc tốt hơn trên mỗi CPU** |
| **Open edX** | Dự án Edly đạt 50.000 user đồng thời bằng cách mở rộng ra 5 pod LMS / 7 node ([edly.io](https://edly.io/academy/project-50k/)) | Cho thấy hướng mở rộng ngang (nhiều worker/pod) — HAUI hiện chỉ có 1 tiến trình API |
| **FastAPI/uvicorn** (TechEmpower Round 23, phần cứng mạnh, nhiều worker) | ~163.894 req/s "single query" (nguồn thứ cấp, [techempower.com](https://www.techempower.com/benchmarks/)) | HAUI 1 worker / 1,5 CPU: 1.700 req/s endpoint rỗng, 100 req/s API thật → chênh lệch chủ yếu do số worker/CPU, middleware và số truy vấn DB mỗi request |

### 5.3 Ngưỡng trải nghiệm người dùng

Theo Nielsen (Nielsen Norman Group): **0,1 s** cảm giác tức thời, **1 s** giữ mạch suy nghĩ, **10 s** giới hạn giữ sự chú ý — quá 10 s cần thanh tiến độ ([nngroup.com](https://www.nngroup.com/videos/3-response-time-limits-interaction-design/)).

| Thao tác HAUI | Đo được | Đánh giá |
|---|---|---|
| API ở tải thường (≤50 user) | 40–390 ms | Trong ngưỡng 1 s ✔ |
| API ở 100–200 user closed-loop | 0,8–1,7 s | Vượt 1 s ⚠ |
| Mở trang qua mạng chậm (50 user) | ~10 s | Ở ngưỡng 10 s ⚠ (bật gzip sẽ còn ~3 s) |
| Chấm video ngắn | 2–4 s | Nằm giữa 1–10 s, có hiển thị trạng thái ✔ |
| Chấm video 60 s | 53–107 s; tới 5 phút khi 6 người nộp | Cần thanh tiến độ / thông báo khi xong (hệ thống đã có polling + notification) ✔ |

---

## 6. Cách chạy lại bộ test

Thư mục: `/home/cuong/haui_loadtest` (bản trên server: `~/loadtest`).

```
haui_loadtest/
├── BAO_CAO_KIEM_THU.md        # báo cáo này
├── jmeter/                     # test plan (.jmx) + accounts.csv
├── scripts/
│   ├── gen_jmx.py              # sinh các file .jmx
│   ├── run_jm.sh               # chạy 1 test JMeter + giám sát tài nguyên + báo cáo HTML
│   ├── run_series.sh           # chạy loạt: fe | be_root | be_login | be_flow | stress | spike | soak
│   ├── e2e_video.py            # test nộp video đầu-cuối
│   ├── monitor.sh              # ghi docker stats, GPU, AI /stats mỗi 2 s
│   ├── summarize_jtl.py        # tổng hợp .jtl → avg/p50/p90/p95/p99/rps
│   └── collect.py              # gom kết quả → results/all_results.json
└── results/                    # kết quả từng test (summary.txt, summary.json, html/ dashboard JMeter, mon/)
```

Ví dụ (chạy trên server, trong `~/loadtest`):

```bash
# FE 10→1000 user, mỗi mức 60 s
bash scripts/run_series.sh fe
# Luồng sinh viên (cần jmeter/tokens.csv dạng "email,token" — đăng nhập trước từng tài khoản)
bash scripts/run_series.sh be_flow
# Một bài lẻ: 200 user, 120 s
bash scripts/run_jm.sh be_flow_200 be_user_flow_tok.jmx -Jhost=127.0.0.1 -Jthreads=200 -Jramp=20 -Jduration=120 -Jtokens=$PWD/jmeter/tokens.csv
# AI trực tiếp: 8 job đồng thời, video đã có trên MinIO
bash scripts/run_jm.sh ai_c8 ai_direct.jmx -Jhost=127.0.0.1 -Jobject_key=submissions/<task_id>/<result_id>.mp4 -Jthreads=8 -Jramp=0 -Jloops=4 -Jduration=3600
# E2E: 10 SV cùng nộp shot1, tổng 20 lượt
python3 scripts/e2e_video.py --base http://127.0.0.1:8000 --video videos/shot1_b.mp4 --task 6ae093a7bf50c057267d76e9 --concurrency 10 --total 20 --out results/e2e/x.csv
```

Dashboard HTML của JMeter cho từng bài: `results/<tên_test>/html/index.html`.

---

## 7. Phụ lục: dữ liệu phát sinh do kiểm thử

- 106 bài nộp video (bản ghi `video_results` + file trên MinIO bucket `videos`, prefix `submissions/6ae093a7bf50c057267d76e9/` và `.../6ae093a7bf50c057267d76ea/`) dưới các tài khoản SV lớp Pickleball (An, Bình, Cường, Đức, Hoa, Huy) — cần dọn nếu hệ thống chuẩn bị dùng thật. 134 job gọi thẳng AI (`job_id` bắt đầu bằng `jm-`) không ghi vào DB.
- Backend bị OOM-kill và tự khởi động lại 1 lần lúc 13:16:41 (bài `be_login_u50`).
- File `jmeter/tokens.csv` chứa token đăng nhập (hạn 7 ngày) — đã xóa trên server; bản trên máy client nên xóa khi không cần chạy lại.

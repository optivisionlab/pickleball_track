# Phương án tối ưu — AI service (nhánh `score-video`) và Backend (nhánh `improve/ai-grading-pipeline`)

Trạng thái: **đề xuất, chưa thực hiện**. Mọi con số dưới đây đo trực tiếp trên server ngày 27/10/2026.

## A. AI service — `pickleball_track`, nhánh `score-video`

### A.0 Hiện trạng đo được

| Đo | shot1 (3 s, 8-bit) | shot2 (60 s, HEVC 10-bit HDR iPhone) |
|---|---|---|
| Tổng thời gian 1 video | 1,2 s | 64,5 s (khi GPU dùng chung) / 48 s (GPU rảnh) |
| Model trên GPU | 0,28 s | **10,8 s (17%)** |
| Âm thanh | 0,25 s | 0,27 s |
| Giải mã + chuẩn bị khung | 0,68 s | **53,4 s (83%)** |
| CPU giải mã của OpenCV | 38 ms/khung | **~260 ms/khung, chiếm ~28 nhân** |

- Nút thắt là **giải mã video HDR 10-bit bằng bản FFmpeg đóng gói trong OpenCV 5.0**: một video đã ăn ~28/48 nhân, nên 2 slot chạy song song không nhanh hơn chạy lần lượt.
- `cv2.setNumThreads`, `OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;N`, giải mã nhiều đoạn song song (thread hoặc process) đều **không cải thiện** (đã thử; pixel/timestamp vẫn giống 100% nhưng tổng CPU không giảm).
- FFmpeg hệ thống (7.1) giải mã cùng video ra BGR: **94 ms CPU/khung, 215 khung/s** (nhanh gấp ~2,1 lần, ít CPU hơn 2,8 lần) — nhưng màu HDR khác OpenCV (lệch trung bình 27/255, đã thử 8 cách quy đổi màu, không khớp).
- `best.pt` (19/10) khớp pipeline huấn luyện v2: OpenCV + giữ nguyên fps gốc. Vì vậy AI hiện tại đúng với cách huấn luyện; đổi bộ giải mã hoặc fps phải kiểm chứng lại độ chính xác.
- **Chạy thêm worker uvicorn sẽ không giúp**: mỗi worker nạp lại toàn bộ model (~3 GB VRAM) và 8 tiến trình tiền xử lý/slot, trong khi nút thắt là CPU giải mã dùng chung.
- Thư mục đang deploy `score_video_block/src` trên đĩa chỉ còn `__pycache__` (mất file `.py`); code trong container khớp 100% commit `150d252` của nhánh `score-video`.

### A.1 Thay đổi an toàn (không đổi dữ liệu đầu vào của model)

| ID | Thay đổi | Kỳ vọng |
|---|---|---|
| AI-1 | Xếp lịch slot theo **video ngắn trước** (biết số khung sau khi đọc metadata) + **dành riêng slot cho video ngắn** (ngưỡng cấu hình, vd ≤ 20 s) | Video ngắn ở giờ cao điểm: p95 105 s → ~3–6 s |
| AI-2 | Tách giới hạn: số job nhận (8 → 16–24) và số lượt tải MinIO đồng thời; `Retry-After` ước tính theo hàng đợi thay vì cố định 10 s | Ít 429 và ít gửi lại vô ích |
| AI-3 | Cấu hình slot theo loại: vd 2 slot video ngắn + 1 slot video dài; benchmark 2/3/4 slot với video 8-bit | Video ngắn: ~90 → dự kiến 120–180 video/phút (cần đo) |
| AI-4 | Endpoint `/stats` thêm hàng đợi theo loại, thời gian chờ slot; log thời gian từng giai đoạn | Dễ giám sát, tinh chỉnh |

### A.2 Thay đổi cần kiểm chứng độ chính xác (đặt sau cờ cấu hình, mặc định TẮT)

| ID | Thay đổi | Kỳ vọng | Kiểm chứng |
|---|---|---|---|
| AI-5 | Giải mã bằng FFmpeg hệ thống qua pipe (giữ nguyên bước resize/normalize hiện tại) | Video HDR 60 s: 48 s → ~20–25 s; 2 slot chạy song song thật → ~3–4 video dài/phút (hiện 1,2) | So sánh hit/miss/điểm trên bộ video có nhãn (dữ liệu huấn luyện) giữa đường cũ và mới |
| AI-6 | Giải mã bằng GPU (NVDEC; cần thêm `video` vào `NVIDIA_DRIVER_CAPABILITIES`) | Gần như không tốn CPU giải mã | Như AI-5 |
| AI-7 | Tối ưu model (SDPA/`torch.compile` cho VideoMAE) | GPU 10,8 s → có thể ~6–8 s | So sánh output số học |

Ghi chú: code huấn luyện v3–v5 (nhánh `baseline_shot1`) lấy mẫu về 30 fps. Nếu sau này deploy model v3+, inference cũng phải lấy mẫu 30 fps — việc này tự nhiên giảm một nửa khối lượng giải mã cho video 60 fps.

### A.3 Cách triển khai

- Tạo git worktree riêng cho nhánh `score-video` (vd `/data/Cuong/pickleball/pickleball_track_score_video`) để không đụng worktree `baseline_shot1` đang dùng huấn luyện; dùng lại `.env` và thư mục weights hiện có.
- Chạy unit test của repo, build image tag mới, đổi container (gián đoạn ~1–2 phút), giữ image cũ `pickleball-track:gpu` để quay lại.

## B. Backend — `haui_sict_fitness_score`, nhánh `improve/ai-grading-pipeline`

| ID | Vấn đề đo được | Thay đổi | Kỳ vọng |
|---|---|---|---|
| BE-1 | Argon2 mặc định `m=64MB,t=3,p=4` → 50 lượt login đồng thời gây OOM | Tham số OWASP (`m=19MB,t=2,p=1`) + semaphore giới hạn số lượt băm đồng thời + tự băm lại mật khẩu cũ khi đăng nhập (`check_needs_rehash`) | Không còn OOM; login ~6 → ~20–40/s |
| BE-2 | Giới hạn 1.024 file → lỗi 500 từ ~1.000 kết nối | `ulimits.nofile: 65536` cho backend, ai-worker; đặt `maxPoolSize` MongoDB | Hết lỗi `Too many open files` |
| BE-3 | `attemptNo` tính bằng đếm khi xin URL → nộp đồng thời bị `E11000` → 500 | Cấp `attemptNo` nguyên tử lúc confirm (bộ đếm `$inc` theo task+user), thử lại khi trùng, trả 409 thay vì 500 | 0 lỗi khi nộp đồng thời |
| BE-4 | Ngưỡng làn 90 s → video 60 s chặn video ngắn | `AI_LANE_THRESHOLD_SEC` → ~20 s (đi cùng AI-1) | Video ngắn không bị chặn |
| BE-5 | ~100 req/s, 1 tiến trình, CPU chạm trần | `uvicorn --workers 3`, `cpus` 1,5 → 3, RAM 768 MB → 1,5 GB (an toàn: outbox claim có lease nguyên tử, Kafka consumer group tự chia partition); thay `BaseHTTPMiddleware` bọc JSON bằng middleware ASGI thuần, giữ nguyên định dạng response | ~100 → ~250–300 req/s |
| BE-6 | `/api/tasks` N+1: mỗi bài tập thêm 2 truy vấn đếm | Gộp thành 1 aggregate `$group` theo taskId | `/api/tasks` nhanh 3–5 lần |
| BE-7 | `/api/tasks` không yêu cầu đăng nhập | Thêm `get_current_user` (cần xác nhận FE không gọi khi chưa đăng nhập) | Bảo mật |
| FE-1 | nginx không gzip, không cache | `gzip on` + `Cache-Control: immutable` cho file có hash (`frontend/nginx.conf`) | 551 KB → 149 KB/lượt |

Lưu ý nhánh backend: đang có thay đổi chưa commit của bạn (trong đó có `video_result_service.py` mà BE-3 cũng cần sửa) và chậm hơn remote 1 commit (`e52c1a7`, sửa đúng các file frontend bạn đang sửa dở).

## C. Kiểm chứng sau khi làm

Chạy lại đúng bộ test cũ (`haui_loadtest`): login 1–50, luồng sinh viên 10–1000, E2E shot1/shot2, giờ cao điểm hỗn hợp, AI trực tiếp; lập bảng trước/sau.

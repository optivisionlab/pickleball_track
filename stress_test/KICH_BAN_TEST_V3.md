# KỊCH BẢN KIỂM THỬ V3 — TỪ CƠ BẢN ĐẾN TỐI ĐA VÀ ĐẾN KHI SẬP (CHỜ DUYỆT)

## 0. Bối cảnh

| | Giá trị |
|---|---|
| Phần cứng | 48 nhân CPU, 123 GB RAM (trống ~96 GB), NVIDIA A30 24 GB, ổ còn 81 GB (đầy 93%) |
| Cấu hình đang chạy (gọi là **H1**) | FE nginx 0,5 CPU / 128 MB · backend 4 worker, 4 CPU / 2 GB · MongoDB 4 CPU / 2 GB · Kafka 1,5 CPU / 1 GB · ai-worker 1 CPU / 512 MB, 6 job · AI 3 instance × 2 slot (không giới hạn CPU), bộ giải mã `opencv` |
| Nhận xét | Hệ thống mới dùng ~14/48 nhân và ~8/123 GB RAM theo giới hạn container. Ở H1 nút thắt đo được là CPU backend (4 nhân), CPU nginx (0,5 nhân) và CPU giải mã video của AI. |
| Dữ liệu test | 300 tài khoản SV test (đã tạo), 1 GV, video `shot1` (3 s, 4 MB), `shot2` (60 s HDR, 100 MB), 40 video thật có nhãn (4,3 GB) |

Kịch bản V3 **thay thế** phần chưa chạy của V2 và giữ lại các bài V2 đã có kết quả.

---

## Giai đoạn 1 — Cơ bản (H1, không tải) · ~20 phút

| ID | Bài | Cách đo | Chỉ số |
|---|---|---|---|
| T1.1 | Kiểm tra sống: FE, BE `/health/ready`, AI `/stats` ×3 instance, LB, MongoDB, Kafka, MinIO | 1 request mỗi dịch vụ | trạng thái, thời gian phản hồi |
| T1.2 | Độ trễ đơn lẻ từng API (~25 endpoint: auth, khóa học, bài tập, bài nộp, sổ điểm, thống kê, thông báo…) | 1 user, mỗi endpoint 200 lần | min / avg / p50 / p95 / p99 |
| T1.3 | Một lượt nộp bài đầu-cuối cho từng loại video: 3 s, 60 s, video thật 30/90/180 s | 1 lượt/loại | thời gian xin URL, upload, xác nhận, chờ hàng đợi, AI chấm, tổng E2E |
| T1.4 | AI đơn lẻ: tách thời gian theo giai đoạn (tải video, giải mã, model GPU, âm thanh) | script đo trong container | giây/giai đoạn, % thời gian |

## Giai đoạn 2 — Tải tăng dần, cấu hình H1 · ~2,5 giờ

Tăng tải theo bậc, mỗi bậc giữ 2 phút. Tìm **điểm gãy** (p95 tăng vọt) và **điểm bão hòa** (thông lượng không tăng nữa).

| ID | Đối tượng | Các bậc tải |
|---|---|---|
| T2.1 | FE: tải trang (12 file) | 100 · 500 · 1.000 · 2.000 · 3.000 user (**đã có kết quả từ V2**) |
| T2.2 | API sinh viên (7 API) | 100 · 500 · 1.000 · 2.000 · 5.000 · 10.000 user |
| T2.3 | Đăng nhập | 10 · 50 · 100 · 200 · 500 lượt đồng thời |
| T2.4 | API giáo viên (sổ điểm, thống kê, bài nộp) | 10 · 50 · 100 · 200 user |
| T2.5 | AI gọi thẳng, video ngắn | 6 · 12 · 24 · 36 · 48 đồng thời |
| T2.6 | AI gọi thẳng, video 60 s | 3 · 6 · 12 đồng thời |
| T2.7 | Upload MinIO (presigned PUT) | 10 · 50 · 100 lượt upload 100 MB đồng thời |

## Giai đoạn 3 — Kịch bản vận hành thực tế, cấu hình H1 · ~3 giờ

Người dùng ảo có thời gian nghỉ ngẫu nhiên, dùng 300 tài khoản riêng biệt.

| ID | Tình huống | Tải | Tiêu chí đạt |
|---|---|---|---|
| T3.1 | Ngày học bình thường | 300 SV duyệt web (nghỉ 3–8 s), 5 GV, 1 bài nộp/20 s (80% ngắn) — 20 phút | API p95 < 500 ms, lỗi 0%, chấm hết |
| T3.2 | Đầu giờ thi | 200 SV đăng nhập trong 30 s → 150 bài nộp trong 5 phút (70% ngắn), hỏi trạng thái mỗi 3 s | đăng nhập p95 < 3 s, 0 lỗi 5xx; đo thời gian chấm hết hàng đợi |
| T3.3 | Hạn chót nộp bài | 300 SV duyệt + 100 bài nộp trong 10 phút, 20% bấm nộp 2 lần | 0 lỗi 500, 0 bài mất |
| T3.4 | Bão hỏi trạng thái | 1.000 SV hỏi `/status` mỗi 2 s + 300 bài nộp trong 60 s | p95 < 300 ms |
| T3.5 | Chấm 40 video thật — so sánh bộ giải mã `opencv` và `ffmpeg` | 40 video nộp trong 60 s, chạy 2 lần | 40/40 chấm xong; video/phút, thời gian xả hàng đợi, độ khớp điểm với nhãn |
| T3.6 | Chạy bền (soak) | tải T3.1 trong 60 phút | RAM/CPU không tăng dần, Kafka lag về 0, không restart |

## Giai đoạn 4 — Cấu hình mở rộng tối đa phần cứng (H2) · ~2 giờ

Tạm nâng giới hạn tài nguyên (sửa biến môi trường compose, **không sửa code**), rồi chạy lại các bài chính để đo hệ thống tăng được bao nhiêu khi có thêm tài nguyên.

| Thành phần | H1 (hiện tại) | H2 (đề xuất) |
|---|---|---|
| FE nginx | 0,5 CPU / 128 MB | 4 CPU / 512 MB |
| Backend | 4 worker · 4 CPU · 2 GB | 16 worker · 16 CPU · 8 GB |
| MongoDB | 4 CPU / 2 GB | 8 CPU / 8 GB |
| Kafka | 1,5 CPU / 1 GB | 4 CPU / 4 GB |
| ai-worker | 1 CPU · 6 job | 2 CPU · 10 job |
| AI | 3 instance × 2 slot, `opencv` | 5 instance × 2 slot, chọn bộ giải mã tốt hơn theo T3.5 |

Chạy lại trên H2: T2.1, T2.2, T2.3, T2.5, T2.6, T3.2, T3.4. Lập bảng **H1 so với H2** (thông lượng, p95, tài nguyên).

## Giai đoạn 5 — Phá giới hạn đến khi sập · ~1,5 giờ

| ID | Bài | Cách làm | Ghi nhận |
|---|---|---|---|
| T5.1 | API quá tải sâu | 10.000 → 20.000 → 30.000 user dồn dập, đến khi hệ thống hoặc máy tạo tải không chịu nổi | mức user lúc lỗi > 50%, thành phần gãy trước |
| T5.2 | Bão đăng nhập | 500 → 1.000 → 2.000 lượt đồng thời | OOM? CPU? thời gian hồi phục |
| T5.3 | Hàng đợi chấm rất sâu | 1.000 bài nộp video ngắn trong 2 phút | Kafka lag, RAM/disk, thời gian chấm hết |
| T5.4 | GPU đến giới hạn | tăng dần số instance AI 5 → 6 → 7 → 8 cho tới khi hết VRAM (OOM) hoặc throughput giảm | số instance tối đa, VRAM, video/phút |
| T5.5 | Tổng hợp cực đại | T3.2 + T3.4 + AI flood + 5.000 user API cùng lúc | thành phần sập trước, lỗi lan truyền |

## Giai đoạn 6 — Sự cố và hồi phục · ~1 giờ

Chạy trên nền tải T3.1 + 20 bài nộp; sau mỗi bài kiểm tra **không mất bài nộp nào** và đo **thời gian hồi phục**.

| ID | Sự cố | Kỳ vọng |
|---|---|---|
| T6.1 | Kill 1 instance AI | LB chuyển instance, job được gửi lại |
| T6.2 | Dừng toàn bộ AI 3 phút | job xếp hàng, chấm khi AI lên lại |
| T6.3 | Restart backend giữa đợt nộp | bài đã xác nhận không mất |
| T6.4 | Dừng Kafka 2 phút | outbox gửi lại hoặc gọi thẳng AI |
| T6.5 | Restart MongoDB | API tự hồi phục |
| T6.6 | Dữ liệu xấu: file không phải video, sai đuôi, quá dài/quá lớn, token giả/hết hạn | trả 4xx đúng, không 500, AI không kẹt |

## Giai đoạn 7 — Từ máy người dùng qua mạng thật · ~20 phút

50 user duyệt web + 3 SV upload video 60 s đồng thời từ máy client (Tailscale).

---

## Chỉ số thu thập (mọi bài)

- **Yêu cầu:** tổng số request, req/s, độ trễ min/avg/p50/p90/p95/p99/max theo từng endpoint, tỷ lệ lỗi theo mã (4xx do dữ liệu tách riêng với lỗi hệ thống 5xx/timeout/mất kết nối).
- **Nộp bài:** thời gian xin URL / upload / xác nhận / chờ hàng đợi / AI chấm / E2E, video/phút, thời gian xả hàng đợi, số bài mất.
- **AI:** busy/pending từng instance, video chờ ngắn/dài, 429, video/phút, thời gian/video, độ khớp điểm với nhãn (T3.5).
- **Tài nguyên (mỗi 2 s):** CPU/RAM từng container và toàn host, GPU util/VRAM/nhiệt độ/công suất, số tiến trình MongoDB, Kafka consumer lag, I/O đĩa và mạng host.
- **Sự kiện:** OOM, restart, container chết, thời gian hồi phục.
- Báo cáo cuối: bảng theo giai đoạn, so sánh với 2 đợt test trước, so sánh H1 và H2, các điểm gãy, khuyến nghị cấu hình.

## An toàn và dọn dẹp

- **Không** làm đầy ổ đĩa: dung lượng upload ước tính ~30 GB (ổ còn 81 GB), có theo dõi; dừng test nếu ổ còn < 25 GB.
- Trước giai đoạn 4 và 5, lưu lại cấu hình H1; sau test **khôi phục H1** (hoặc giữ H2 nếu bạn muốn).
- Sau test: xóa bài nộp test, file MinIO test, 300 tài khoản test.
- Máy tạo tải (JMeter, script) chạy trên cùng server nên cũng dùng CPU; tôi ghi lại CPU của nó để tách riêng.

## Tổng thời gian dự kiến

| Giai đoạn | Thời gian |
|---|---|
| 1 Cơ bản | 0,3 giờ |
| 2 Tải tăng dần H1 | 2,5 giờ |
| 3 Vận hành thực tế H1 | 3 giờ |
| 4 Mở rộng H2 | 2 giờ |
| 5 Đến khi sập | 1,5 giờ |
| 6 Sự cố | 1 giờ |
| 7 Từ client | 0,3 giờ |
| **Tổng** | **~10,5 giờ** |

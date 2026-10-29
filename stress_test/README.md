# Kiểm thử hiệu năng HaUI Fitness (V3)

Kết quả cuối cùng: mỗi kịch bản và mỗi mức tải chỉ giữ một lần chạy sạch tốt nhất (GPU không dùng chung, cấu hình đã sửa lỗi). Mọi số trong báo cáo lấy từ thư mục `results/`.

| Đường dẫn | Nội dung |
| --- | --- |
| `BAO_CAO_HIEU_NANG_V3.docx`, `BAO_CAO_HIEU_NANG_V3.md` | Báo cáo đầy đủ (bản Markdown dùng hình trong `bao_cao_v3_media/`) |
| `KICH_BAN_TEST_V3.md` | Thiết kế bộ kiểm thử GD1–GD5 |
| `results/` | Tóm tắt từng bài: `summary.txt`/`summary.json`, tham số, số liệu giám sát `mon/` (đã bỏ file `.jtl` thô) |
| `results/v2/` | Các kịch bản của bộ mô phỏng người dùng (GD1 E2E, GD3, T5.3, T5.5) |
| `results/v3/` | Độ trễ endpoint GD1, upload, đầu đo độc lập T3.4, log GD4b + GD5 (`chain6.log`) |
| `scripts/`, `jmeter/` | Script chạy lại: `scripts/run_v3.sh <gd1|gd2|gd3|gd4|gd4b|gd5|aiscale|h1|h2>` |

Mật khẩu tài khoản thật đọc từ biến môi trường `HAUI_ADMIN_PASSWORD`, `HAUI_TEACHER_PASSWORD`, `HAUI_STUDENT_PASSWORD`.
Bản đầy đủ (kể cả `.jtl` và các lần chạy bị loại) nằm trên server ở `~/loadtest/results`.

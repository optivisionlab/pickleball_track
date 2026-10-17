# 📥 YouTube Batch Downloader

Tool tải **hàng loạt video YouTube chất lượng cao** (tối đa 4K/8K, 60fps, HDR) từ file CSV, có **giao diện** chạy trên trình duyệt. Tool chạy hoàn toàn trên máy của bạn và không cần tài khoản.

- Kéo thả file CSV, hoặc dán link trực tiếp
- Chọn chất lượng, định dạng (mp4/mkv) và số video tải song song
- Theo dõi tiến độ, tốc độ, thời gian còn lại của từng video
- Có thể dừng giữa chừng và tải tiếp; video đã tải sẽ được bỏ qua
- Tự thử lại khi lỗi mạng; xuất báo cáo CSV

---

## 1. Cài đặt: 1 lệnh

> Dành cho **Ubuntu / Linux / macOS**, cần có sẵn `git` và `python3` (≥ 3.10). Mở terminal bằng `Ctrl + Alt + T`.

```bash
git clone -b ytb-download <địa-chỉ-repo> && cd <tên-repo>
bash downloadyt/install.sh
```

Thấy dòng **`✔ Cài xong!`** là thành công, giao diện sẽ tự mở trên trình duyệt.
Có thể máy sẽ hỏi **mật khẩu sudo**, đó là bước cài `ffmpeg` hoặc `python3-venv` nếu máy chưa có.

`install.sh` tự làm hết các bước sau:

| Bước | Chi tiết |
|---|---|
| Môi trường Python riêng | tạo `downloadyt/.venv`, không ảnh hưởng Python của máy (tự cài `python3-venv` nếu thiếu) |
| Thư viện | cài `yt-dlp` bản mới nhất, kèm **deno** (JavaScript runtime mà YouTube cần để trả về bản 4K/1080p), không cần sudo |
| `ffmpeg` | tự cài nếu chưa có (apt / dnf / pacman / brew). Cần để **ghép hình + tiếng** |
| Lệnh `ytdown` | tạo trong `~/.local/bin` và tự thêm vào PATH (`~/.bashrc`, `~/.zshrc`) |

Chạy lại `install.sh` bất cứ lúc nào cũng an toàn. Chỉ muốn cài mà không mở tool thì dùng `bash downloadyt/install.sh --no-launch`.

---

## 2. Mở tool

Mở **terminal mới** (để PATH có lệnh `ytdown`) và gõ:
```bash
ytdown
```
Trình duyệt sẽ tự mở giao diện tại **http://127.0.0.1:8765**. Nếu không tự mở, copy địa chỉ này vào trình duyệt.

> ⚠ **Giữ cửa sổ terminal mở** trong lúc dùng tool. Tắt terminal thì tool cũng tắt.
> Lỡ đóng tab trình duyệt? Gõ lại `ytdown` để mở lại, video đang tải không bị ảnh hưởng.

**Tắt tool:** bấm nút **"Tắt tool"** trên giao diện, hoặc nhấn `Ctrl + C` ở terminal.

---

## 3. Sử dụng

1. **Thêm video:** kéo thả file CSV vào cửa sổ tool (hoặc bấm để chọn file), hoặc dán link vào ô bên phải rồi bấm **"Thêm vào danh sách"**.
2. **Thư mục lưu:** bấm **"Chọn..."** để đổi. Mặc định video được lưu vào thư mục `downloads/` bên trong thư mục tool.
3. **Chất lượng tối đa:** mặc định là 4K. Nếu video không có mức đó, tool tự lấy mức cao nhất có sẵn.
4. Bấm **▶ Bắt đầu tải**.
5. Tải xong, bấm **▶** để xem video, **📁** để mở thư mục chứa video, hoặc **⬇ Tải báo cáo CSV** để lấy báo cáo.

Các cài đặt được lưu tự động cho lần sau.

### Định dạng file CSV

Chỉ cần **một cột link**:
```csv
url
https://www.youtube.com/watch?v=xxxxxxxxxxx
https://youtu.be/yyyyyyyyyyy
```

Có thể thêm các cột tùy chọn:
```csv
url,name,quality,folder
https://www.youtube.com/watch?v=xxxxxxxxxxx,tran_dau_1,2160,pickleball
https://youtu.be/yyyyyyyyyyy,,1080,
```

| Cột | Bắt buộc | Ý nghĩa |
|---|---|---|
| `url` (hoặc `link`, `video`) | ✔ | link YouTube |
| `name` | | tên file mong muốn (để trống thì dùng tiêu đề video) |
| `quality` | | độ phân giải tối đa riêng cho video đó: `2160`, `1080`, `4k`... |
| `folder` | | lưu vào thư mục con này |

- File có thể phân cách bằng `,`, `;` hoặc tab. File xuất từ Excel hay Google Sheets đều dùng được.
- File **không có tiêu đề cột** cũng được: tool tự tìm mọi link YouTube có trong file.
- Link trùng nhau sẽ tự bị bỏ qua.
- Xem file mẫu `links_mau.csv`.

### Các tùy chọn

| Tùy chọn | Ý nghĩa |
|---|---|
| Chất lượng tối đa | Cao nhất có thể / 8K / 4K / 2K / 1080p / 720p ... |
| Định dạng | `mp4` (dễ mở nhất), `mkv`, `webm` |
| Tải song song | số video tải cùng lúc (2–3 là hợp lý) |
| Bỏ qua video đã tải | bỏ tích nếu muốn tải lại video đã tải trước đó |
| Chỉ hình (không tiếng) | chỉ tải phần hình, không có tiếng |
| *Nâng cao →* Phụ đề | tải phụ đề, ví dụ `vi,en` |
| *Nâng cao →* Cookies từ trình duyệt | dùng cho video giới hạn tuổi/thành viên, hoặc khi bị YouTube chặn |
| *Nâng cao →* Tải cả playlist | link playlist sẽ tải toàn bộ video trong playlist |

---

## 4. Cập nhật

YouTube thay đổi thường xuyên. Khi tool **tải lỗi hàng loạt**, hoặc khoảng **mỗi tháng một lần**, hãy chạy:
```bash
ytdown --update
```

Kiểm tra máy đã đủ `ffmpeg` và JavaScript runtime chưa: `ytdown --check`.

---

## 5. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ytdown: command not found` | Mở terminal mới. Nếu vẫn lỗi, chạy trực tiếp `đường/dẫn/tới/downloadyt/ytdown` |
| Video tải về không có tiếng / chất lượng thấp | Chạy `ytdown --check`; nếu thiếu thì chạy lại `bash downloadyt/install.sh` |
| `Sign in to confirm you're not a bot` | *Nâng cao → Cookies từ trình duyệt*: chọn trình duyệt đang đăng nhập YouTube |
| `Sign in to confirm your age` / video thành viên | Như trên: dùng cookies từ trình duyệt |
| `Video unavailable` / `Private video` | Video đã bị xóa hoặc để riêng tư, không tải được |
| Tải lỗi hàng loạt video vốn vẫn xem được | `ytdown --update` (mục 4) |
| Bấm "Chọn..." không hiện hộp thoại | Gõ đường dẫn thư mục trực tiếp vào ô "Thư mục lưu" |

---

## 6. Dùng bằng dòng lệnh (không cần giao diện)

```bash
ytdown --cli links.csv                       # tối đa 4K, lưu vào ./downloads
ytdown --cli links.csv -o videos -q 1080 -j 3
ytdown --cli --help                          # xem tất cả tùy chọn
```

---

## Cấu trúc thư mục

```
downloadyt/
├── README.md          ← hướng dẫn này
├── install.sh         ← cài đặt bằng 1 lệnh
├── ytdown             ← lệnh mở tool (--cli, --update, --check)
├── ytgui.py           ← giao diện
├── tooldownyt.py      ← phần tải video (dùng được bằng dòng lệnh)
├── requirements.txt   ← thư viện Python cần cài
├── links_mau.csv      ← file CSV mẫu
└── downloads/         ← video tải về (tự tạo)
```

> `.venv/` và `downloads/` đã nằm trong `.gitignore` nên không bị commit lên git.

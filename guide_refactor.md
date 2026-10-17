# TASK: Refactor AI Inference API to Production-Ready Architecture

Bạn là Senior AI Engineer, ML Engineer và Backend Architect. Hãy refactor toàn bộ repository hiện tại thành một hệ thống **AI Inference API có kiến trúc sạch, dễ bảo trì, dễ kiểm thử và sẵn sàng tích hợp với backend bên ngoài**.

## 1. Bối cảnh dự án

Đây là một API phục vụ suy luận mô hình AI. Một hệ thống backend khác sẽ gọi API này, gửi dữ liệu đầu vào và nhận kết quả dự đoán.

Mô hình hiện tại sử dụng các pretrained model và kiến trúc học sâu do dự án tự xây dựng:

- VideoMAE: trích xuất đặc trưng video.
- BEATs: trích xuất đặc trưng âm thanh, nếu được sử dụng trong phiên bản model hiện tại.
- Custom feature projection / feature extraction blocks.
- Feature fusion modules để kết hợp các đặc trưng.
- Transformer hoặc Temporal Transformer để học quan hệ giữa các đặc trưng.
- Classification head để dự đoán các nhãn.
- Trained checkpoint/weights được tạo từ quá trình training.

Bài toán phân loại hiện tại có các nhãn:

- `no_event`: không có sự kiện.
- `hit`: đánh trúng.
- `miss`: đánh trượt.

Hãy xác minh nhãn, kiến trúc và preprocessing trực tiếp từ source code, checkpoint và cấu hình hiện có. Không tự giả định rằng mô hình đang dùng audio nếu phiên bản hiện tại đã loại bỏ audio.

## 2. Mục tiêu refactor

Tổ chức code theo các nguyên tắc:

- Separation of concerns: tách biệt API, business logic, model architecture, preprocessing và inference.
- Single responsibility: mỗi module chỉ chịu trách nhiệm cho một nhiệm vụ rõ ràng.
- Dependency injection ở những nơi phù hợp.
- Type hints đầy đủ và Pydantic schemas cho API.
- Logging có cấu trúc, thông báo lỗi rõ ràng.
- Configuration quản lý bằng environment variables.
- Có unit tests, integration tests và hướng dẫn chạy.
- Giữ nguyên hành vi và kiến trúc mô hình hiện tại, trừ khi có lý do kỹ thuật rõ ràng và được giải thích.
- Không thay đổi logic training, model weights, preprocessing hoặc output semantics một cách tùy tiện.

## 3. Khảo sát repository trước khi sửa

Trước khi viết code:

1. Đọc toàn bộ cây thư mục và xác định entry point của API.
2. Xác định framework backend hiện tại, ví dụ FastAPI.
3. Tìm tất cả định nghĩa model, pretrained model loaders, feature fusion, Transformer, classification head và checkpoint loaders.
4. Xác định cách VideoMAE và BEATs được khởi tạo, cấu hình và gọi.
5. Xác định preprocessing video/audio, sampling frames, sampling rate, normalization, tensor shape và device handling.
6. Xác định mapping nhãn, output logits/probabilities và cách chuyển thành kết quả nghiệp vụ.
7. Xác định các file training, inference, dataset, evaluation và checkpoint.
8. Kiểm tra các đường dẫn tuyệt đối, secrets, dependency imports và code trùng lặp.
9. Xác định request/response contract mà backend bên ngoài đang sử dụng.

Tạo bản tóm tắt kiến trúc hiện tại trước khi thay đổi. Không xóa hoặc ghi đè các file quan trọng trước khi hiểu đầy đủ các dependency của chúng.

## 4. Kiến trúc thư mục mục tiêu

Điều chỉnh cấu trúc dưới đây theo repository thực tế, không tạo file rỗng chỉ để làm đẹp kiến trúc.

```text
project/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   └── inference.py
│   │   └── schemas/
│   │       ├── request.py
│   │       └── response.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   ├── services/
│   │   └── inference_service.py
│   ├── models/
│   │   ├── architecture.py
│   │   ├── components/
│   │   │   ├── video_encoder.py
│   │   │   ├── audio_encoder.py
│   │   │   ├── fusion.py
│   │   │   ├── transformer.py
│   │   │   └── classifier.py
│   │   └── model_loader.py
│   ├── pipelines/
│   │   └── preprocessing.py
│   └── utils/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── configs/
├── checkpoints/
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
└── README.md
```

Lưu ý: tên module phải phản ánh code thực tế. Nếu BEATs không còn được sử dụng, không tạo audio encoder giả hoặc giữ module không cần thiết.

## 5. Quy định đối với model và weights

Đây là phần quan trọng nhất.

- Tách model architecture khỏi model loading và API.
- Khởi tạo model một lần trong application lifespan hoặc cơ chế quản lý vòng đời tương đương.
- Load pretrained VideoMAE/BEATs và trained checkpoint đúng theo cấu hình hiện tại.
- Không tải lại weights cho mỗi request.
- Không train hoặc fine-tune trong quá trình xử lý request.
- Đặt model ở chế độ `eval()`.
- Sử dụng `torch.inference_mode()` cho inference khi phù hợp.
- Xác định device thông qua cấu hình và khả năng thực tế: CUDA, CPU hoặc accelerator được hỗ trợ.
- Xử lý đúng dtype, tensor shapes, attention masks và batch dimensions.
- Giữ nguyên mapping nhãn và thứ tự đầu ra tương thích với checkpoint.
- Không dùng `strict=False` để che giấu lỗi checkpoint không tương thích. Nếu cần dùng, phải có lý do rõ ràng và báo cáo các key thiếu/thừa.
- Không tự thay đổi kiến trúc model để làm cho checkpoint load được.
- Không hard-code đường dẫn checkpoint, pretrained weights hoặc thư mục dữ liệu.
- Không commit weights lớn hoặc secrets lên Git nếu repository đang quản lý chúng bên ngoài.

Tách riêng ba khái niệm:

1. Pretrained weights của các backbone.
2. Trained checkpoint của mô hình kết hợp.
3. Cấu hình cần thiết để tái tạo chính xác kiến trúc và preprocessing.

Nếu checkpoint hiện tại chứa cả backbone và fusion/classifier weights, phải giữ đúng logic loading đó.

## 6. Thiết kế API

Tách API routes khỏi inference logic.

Kiểm tra và giữ tương thích endpoint, HTTP method, request schema và response schema hiện tại nếu đã được hệ thống bên ngoài sử dụng.

Nếu chưa có contract ổn định, đề xuất endpoint như:

- `GET /health`: kiểm tra service.
- `GET /ready`: kiểm tra model đã load và sẵn sàng inference.
- `POST /predict`: nhận dữ liệu và trả kết quả dự đoán.

Request có thể nhận video upload hoặc tham chiếu video tùy contract hiện tại. Không tự đổi sang URL, multipart upload hoặc base64 nếu chưa xác minh cách backend đang tích hợp.

Response cần thể hiện được các thông tin phù hợp với contract hiện có, ví dụ:

- Nhãn dự đoán.
- Xác suất từng lớp nếu model thực sự hỗ trợ việc diễn giải đó.
- Trạng thái xử lý.
- Request ID nếu phù hợp.

Không tự bịa thêm trường bắt buộc làm hỏng client hiện tại.

Phân biệt rõ:

- `no_event`: model dự đoán không có sự kiện.
- `hit`: model dự đoán đánh trúng.
- `miss`: model dự đoán đánh trượt.
- HTTP error: request không hợp lệ hoặc service gặp lỗi.

Không biến lỗi xử lý video thành nhãn `no_event`.

## 7. Preprocessing và inference pipeline

Tạo pipeline rõ ràng, có thể kiểm thử độc lập:

1. Validate input.
2. Đọc video hoặc dữ liệu đã được cung cấp.
3. Preprocess theo đúng pipeline khi training.
4. Trích xuất đặc trưng bằng backbone tương ứng.
5. Fusion các đặc trưng theo đúng kiến trúc đã huấn luyện.
6. Chạy Transformer và classification head.
7. Chuyển logits thành probabilities nếu phù hợp.
8. Ánh xạ index thành label.
9. Serialize response.

Không thay đổi frame sampling, normalization, temporal window, audio sampling hoặc feature dimension tùy tiện vì những thay đổi này có thể khiến inference lệch với training.

Nếu API phải nhận video dài nhưng model hiện tại chỉ hỗ trợ clip ngắn, giữ nguyên giới hạn đã xác minh và báo lỗi validation rõ ràng. Không tự thêm event spotting cho video dài nếu tính năng đó chưa được triển khai.

## 8. Configuration và bảo mật

- Tập trung cấu hình vào một module.
- Sử dụng environment variables cho đường dẫn, device, model configuration, giới hạn upload và các thông số triển khai.
- Tạo `.env.example` chỉ chứa giá trị minh họa không nhạy cảm.
- Không ghi API keys, passwords, tokens hoặc nội dung nhạy cảm vào logs.
- Kiểm tra loại file, dung lượng và input trước khi xử lý.
- Dọn dẹp file tạm an toàn kể cả khi inference thất bại.
- Không tự tải model từ nguồn không đáng tin cậy.
- Không sử dụng pickle/checkpoint từ nguồn không đáng tin cậy.

## 9. Performance và concurrency

- Không load model theo từng request.
- Không dùng chung tài nguyên GPU theo cách gây race condition.
- Đánh giá việc giới hạn số inference đồng thời và queue nếu cần.
- Tránh chặn event loop FastAPI bằng tác vụ inference nặng.
- Không tạo multiprocessing hoặc batching phức tạp nếu chưa có nhu cầu thực tế.
- Theo dõi latency, lỗi inference và mức sử dụng tài nguyên bằng logging phù hợp.
- Đảm bảo shutdown giải phóng tài nguyên đúng cách.

## 10. Testing bắt buộc

Viết hoặc cập nhật các test phù hợp với code thực tế:

### Unit tests
- Label mapping.
- Input validation.
- Preprocessing và tensor shapes.
- Model loader.
- Output conversion.
- Exception handling.

### Integration tests
- API khởi động với cấu hình test.
- Health/readiness endpoint.
- Request không hợp lệ trả HTTP status phù hợp.
- Predict trả response đúng schema.
- Checkpoint load thành công với model architecture tương ứng.
- Model inference chạy được trên một fixture nhỏ hoặc mock có kiểm soát.

### Regression tests
Nếu có dữ liệu và checkpoint mẫu hợp lệ, so sánh output của code cũ và code mới trên cùng input.

Đối với refactor thuần túy, output phải giữ tương đương trong sai số số học hợp lý. Không khẳng định accuracy hoặc macro-F1 được cải thiện nếu chưa thực hiện evaluation thực tế.

Nếu không có checkpoint, dữ liệu hoặc accelerator cần thiết, ghi rõ phần nào chưa thể xác minh; không giả lập kết quả thành công.

## 11. Kiểm tra sau refactor

Thực hiện các bước sau nếu môi trường hỗ trợ:

1. Kiểm tra import và cú pháp.
2. Chạy formatter và linter đã có trong repository.
3. Chạy unit tests.
4. Chạy integration tests.
5. Kiểm tra model checkpoint loading.
6. Chạy inference trên một mẫu thực tế nếu có dữ liệu.
7. Kiểm tra request/response của API.
8. Tìm các import cũ, đường dẫn hard-code và module không còn sử dụng.
9. Kiểm tra diff để bảo đảm không xóa nhầm training scripts, dataset code, checkpoints hoặc cấu hình quan trọng.
10. Cập nhật README với cách cài đặt, cấu hình, khởi chạy API, gọi API và chạy test.

Không cài thêm dependency hoặc thay đổi phiên bản thư viện lớn nếu chưa có lý do kỹ thuật.

## 12. Quy tắc thực thi

- Làm việc trực tiếp trên repository hiện tại.
- Không chỉ trả về hướng dẫn hoặc code mẫu: hãy thực hiện refactor.
- Ưu tiên các thay đổi nhỏ, có thể kiểm tra và rollback.
- Giữ nguyên API contract hiện có, trừ khi được yêu cầu thay đổi.
- Không xóa code training hoặc evaluation chỉ vì nó không được import bởi API.
- Không sửa thuật toán hoặc hyperparameters training trong task refactor API.
- Không tạo các abstraction không cần thiết.
- Không che giấu lỗi bằng cách bỏ qua exceptions hoặc dùng fallback không an toàn.
- Không tuyên bố test pass nếu chưa chạy và quan sát kết quả.

## 13. Báo cáo cuối cùng

Sau khi hoàn tất, cung cấp:

1. Kiến trúc cũ và các vấn đề đã phát hiện.
2. Cây thư mục mới.
3. Danh sách file tạo, sửa, di chuyển hoặc xóa.
4. Giải thích trách nhiệm của từng module chính.
5. Cách model được khởi tạo và weights được load.
6. Request/response contract của API.
7. Các lệnh cài đặt, chạy API và chạy test.
8. Kết quả từng lệnh kiểm thử, kèm lỗi nếu có.
9. Những phần chưa thể xác minh và lý do.
10. Các rủi ro tương thích cần lưu ý trước khi deploy.

Bắt đầu bằng việc khảo sát repository và xác định dependency graph. Sau đó mới lập kế hoạch refactor, triển khai từng phần và kiểm thử. Mục tiêu là một AI Inference API có cấu trúc rõ ràng, giữ nguyên hành vi mô hình đã huấn luyện và có thể tích hợp ổn định với hệ thống backend bên ngoài.
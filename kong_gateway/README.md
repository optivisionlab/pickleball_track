# Kong API Gateway (Hybrid Mode)

Hệ thống **Kong API Gateway** triển khai theo kiến trúc **Hybrid Mode (Control Plane & Data Plane tách biệt)** bằng Docker Compose.

---

## ⚡ Khởi động nhanh (Quick Start)

### 1. Sinh chứng chỉ mTLS cho Cluster
- **Windows**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\generate-certs.ps1
  ```
- **Linux**:
  ```bash
  chmod +x scripts/generate-certs.sh
  ./scripts/generate-certs.sh
  ```

### 2. Khởi chạy hệ thống
```bash
docker compose up -d
```

---

## 🌐 Các cổng dịch vụ

| Cổng (Port) | Dịch vụ | Mục đích |
| :--- | :--- | :--- |
| **`:8002`** | **Kong Manager** | Giao diện Web UI quản trị (Tạo Service, Route, Plugin) |
| **`:8001`** | **Admin API** | REST API cấu hình hệ thống cho Control Plane |
| **`:8000`** | **Proxy HTTP** | Cổng đón nhận traffic chính từ Client / Frontend |
| **`:8443`** | **Proxy HTTPS** | Cổng đón nhận traffic bảo mật SSL/TLS |
| **`:8005`** | **Cluster mTLS** | Kênh gRPC nội bộ đồng bộ cấu hình giữa CP và DP |

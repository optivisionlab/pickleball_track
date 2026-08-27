#!/usr/bin/env bash
# ==============================================================================
# Script sinh chứng chỉ mTLS cho cụm Kong Hybrid (Linux Server)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="${SCRIPT_DIR}/../certs"
CERT_FILE="${CERT_DIR}/cluster.crt"
KEY_FILE="${CERT_DIR}/cluster.key"

mkdir -p "${CERT_DIR}"

echo "🔐 Đang tạo chứng chỉ mTLS cho Kong Cluster..."

openssl req -new -x509 -nodes -days 3650 -subj "/CN=kong_clustering" -keyout "${KEY_FILE}" -out "${CERT_FILE}"

chmod 600 "${KEY_FILE}"
chmod 644 "${CERT_FILE}"

echo "✅ Tạo chứng chỉ mTLS thành công!"
echo "   - Certificate: ${CERT_FILE}"
echo "   - Private Key: ${KEY_FILE}"

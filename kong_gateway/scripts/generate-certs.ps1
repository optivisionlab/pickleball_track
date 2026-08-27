# ==============================================================================
# Script sinh chung chi mTLS cho cum Kong Hybrid (Control Plane <-> Data Plane)
# ==============================================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$certDir = Join-Path $scriptDir "..\certs"
$certFile = Join-Path $certDir "cluster.crt"
$keyFile = Join-Path $certDir "cluster.key"

if (-not (Test-Path $certDir)) {
    New-Item -ItemType Directory -Path $certDir -Force | Out-Null
}

Write-Host "Creating mTLS certificates for Kong Cluster..." -ForegroundColor Cyan

# OpenSSL command to generate self-signed cert for Kong Hybrid Cluster
openssl req -new -x509 -nodes -days 3650 -subj "/CN=kong_clustering" -keyout "$keyFile" -out "$certFile"

if ((Test-Path $certFile) -and (Test-Path $keyFile)) {
    Write-Host "SUCCESS: mTLS certificates created successfully!" -ForegroundColor Green
    Write-Host "   - Certificate: $certFile" -ForegroundColor Gray
    Write-Host "   - Private Key: $keyFile" -ForegroundColor Gray
} else {
    Write-Host "ERROR: Failed to generate certificates. Please check OpenSSL." -ForegroundColor Red
}

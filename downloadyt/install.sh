#!/usr/bin/env bash
# Cài YouTube Batch Downloader bằng 1 lệnh, sau đó tự mở giao diện:
#
#   bash downloadyt/install.sh            # cài + mở tool
#   bash downloadyt/install.sh --no-launch
#
# Lần sau chỉ cần gõ: ytdown
# Chạy lại script bất cứ lúc nào cũng an toàn (cập nhật yt-dlp, sửa lệnh ytdown).
set -euo pipefail

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV="$DIR/.venv"
BIN_DIR="$HOME/.local/bin"
LAUNCH=1
[[ "${1:-}" == "--no-launch" ]] && LAUNCH=0

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m⚠\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m✘\033[0m  %s\n' "$*" >&2; exit 1; }

# Cài gói hệ thống (cần sudo); trả về 1 nếu không có trình quản lý gói quen thuộc.
install_system_pkg() {
  if command -v apt-get >/dev/null; then sudo apt-get update -qq && sudo apt-get install -y "$@"
  elif command -v dnf >/dev/null; then sudo dnf install -y "$@"
  elif command -v pacman >/dev/null; then sudo pacman -S --needed --noconfirm "$@"
  elif command -v brew >/dev/null; then brew install "$@"
  else return 1
  fi
}

# ---------------------------------------------------------------- Python >= 3.10
PYTHON=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PYTHON="$(command -v "$cand")"; break
  fi
done
[[ -n "$PYTHON" ]] || die "Cần Python >= 3.10 (Ubuntu: sudo apt install python3)"
say "Dùng $("$PYTHON" --version) tại $PYTHON"

# ---------------------------------------------------------------- venv + yt-dlp
if [[ ! -x "$VENV/bin/python" ]]; then
  say "Tạo môi trường Python riêng: $VENV"
  if ! "$PYTHON" -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    # Ubuntu/Debian tách module venv ra gói riêng.
    pyver="$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    warn "Thiếu module venv, cài python${pyver}-venv (cần mật khẩu sudo)"
    install_system_pkg "python${pyver}-venv" || die "Không tạo được venv. Cài gói python3-venv rồi chạy lại."
    "$PYTHON" -m venv "$VENV"
  fi
fi
say "Cài/cập nhật yt-dlp + deno (JS runtime cho YouTube)"
"$VENV/bin/python" -m pip install -q --disable-pip-version-check -U pip
"$VENV/bin/python" -m pip install -q --disable-pip-version-check -U -r "$DIR/requirements.txt"

# ---------------------------------------------------------------- ffmpeg (ghép hình + tiếng)
if ! command -v ffmpeg >/dev/null; then
  warn "Chưa có ffmpeg — cần để ghép hình + tiếng. Đang cài (có thể hỏi mật khẩu sudo)..."
  install_system_pkg ffmpeg || warn "Không tự cài được ffmpeg. Hãy cài thủ công (https://ffmpeg.org/download.html) rồi chạy lại."
fi

# ---------------------------------------------------------------- lệnh ytdown
chmod +x "$DIR/ytdown"
mkdir -p "$BIN_DIR"
ln -sf "$DIR/ytdown" "$BIN_DIR/ytdown"
say "Đã tạo lệnh: $BIN_DIR/ytdown"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    line='export PATH="$HOME/.local/bin:$PATH"  # ytdown'
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
      if [[ -f "$rc" || "$rc" == "$HOME/.bashrc" ]] && ! grep -qF "$line" "$rc" 2>/dev/null; then
        printf '\n%s\n' "$line" >> "$rc"
        say "Đã thêm ~/.local/bin vào PATH trong $rc (mở terminal mới để dùng lệnh ytdown)"
      fi
    done
    ;;
esac

# ---------------------------------------------------------------- kiểm tra
echo "   yt-dlp $("$VENV/bin/python" -c 'import yt_dlp; print(yt_dlp.version.__version__)')"
"$DIR/ytdown" --check || warn "Tool vẫn chạy được nhưng có thể thiếu chất lượng cao / không ghép được tiếng (xem cảnh báo trên)."

printf '\033[1;32m✔\033[0m  Cài xong! Lần sau chỉ cần gõ: ytdown\n'
if (( LAUNCH )); then
  exec "$DIR/ytdown"
fi

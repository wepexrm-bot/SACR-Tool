#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  SACR Tool — One-Click Installer"
echo "========================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Install Python from python.org first."
    exit 1
fi

# Install pipx if missing
if ! command -v pipx &> /dev/null; then
    echo "[1/3] Installing pipx..."
    python3 -m pip install --user pipx -q
fi

# Install the package
echo "[2/3] Installing SACR Tool..."
pipx install git+https://github.com/wepexrm-bot/SACR-Tool.git --force

# Add to PATH automatically
echo "[3/3] Adding to PATH..."
pipx ensurepath &> /dev/null || true

# Create symlink in /usr/local/bin for global access
if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ] || command -v sudo &> /dev/null; then
    SRC="$HOME/.local/bin/sacr_cli"
    if [ -f "$SRC" ]; then
        sudo ln -sf "$SRC" /usr/local/bin/sacr_cli 2>/dev/null || true
        echo "  Symlinked to /usr/local/bin/sacr_cli (global)"
    fi
fi

# Source shell rc so it works immediately
SHELL_RC="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
fi
if [ -f "$SHELL_RC" ]; then
    source "$SHELL_RC" 2>/dev/null || true
fi

export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "========================================================"
echo "  INSTALLATION COMPLETE!"
echo "========================================================"
echo ""
echo "  Try it now:  sacr_cli --version"
echo "  Or open a new terminal and type:  sacr_cli"
echo ""
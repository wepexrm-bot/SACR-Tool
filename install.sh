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

# Install
echo "[1/2] Installing SACR Tool..."
pip3 install git+https://github.com/wepexrm-bot/SACR-Tool.git

# Detect scripts dir and add to PATH via shell profile
echo "[2/2] Adding SACR Tool to PATH..."
SCRIPTS_DIR=$(python3 -c "import site; print(site.USER_BASE)" 2>/dev/null)/bin
if [ ! -d "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(python3 -c "import sys; print(sys.base_exec_prefix)" 2>/dev/null)/bin
fi

SHELL_RC="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "sacr_cli" "$SHELL_RC" 2>/dev/null; then
    echo "export PATH=\"\$PATH:$SCRIPTS_DIR\"  # SACR Tool" >> "$SHELL_RC"
fi

echo ""
echo "========================================================"
echo "  INSTALLATION COMPLETE!"
echo "========================================================"
echo ""
echo "  Run this to load it now:  source $SHELL_RC"
echo "  Then type:                sacr_cli"
echo ""
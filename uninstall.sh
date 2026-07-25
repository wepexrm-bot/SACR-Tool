#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  SACR Tool — Uninstaller"
echo "========================================================"
echo ""

# Remove via pipx or pip
if command -v pipx &> /dev/null; then
    pipx uninstall sacr-tool 2>/dev/null || true
fi
pip3 uninstall sacr-tool -y 2>/dev/null || true

# Remove from PATH in shell rc
SHELL_RC="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [ -f "$SHELL_RC" ]; then
    sed -i '/# SACR Tool/d' "$SHELL_RC" 2>/dev/null || true
fi

echo ""
echo "========================================================"
echo "  UNINSTALL COMPLETE!"
echo "========================================================"
echo ""
echo "  Close and reopen your terminal to refresh PATH."
echo ""
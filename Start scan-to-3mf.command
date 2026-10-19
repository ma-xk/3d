#!/bin/bash
# Double-click this file in Finder to start scan-to-3mf in your browser.
# Close this Terminal window (or press Ctrl+C) to stop it.

cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

pause() { echo; read -r -p "Press Enter to close this window. " _; }

if ! command -v uv >/dev/null 2>&1; then
  echo "scan-to-3mf runs with uv, a small Python project runner from Astral (https://astral.sh/uv)."
  read -r -p "uv is not installed. Install it now? [y/N] " answer
  case "$answer" in
    [yY]*)
      if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "Installing uv failed."; pause; exit 1
      fi
      export PATH="$HOME/.local/bin:$PATH"
      ;;
    *)
      echo "Install uv, then double-click this file again."; pause; exit 1
      ;;
  esac
fi

echo "Starting scan-to-3mf. The first start downloads Python and the packages it needs (about 150 MB);"
echo "later starts take a few seconds. Your browser opens when it is ready."
echo
uv run --frozen scan-to-3mf serve
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "scan-to-3mf stopped with an error (exit code $status)."
  pause
fi
exit $status

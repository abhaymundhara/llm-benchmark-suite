#!/usr/bin/env bash
# Launch the Streamlit benchmark dashboard.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

if ! command -v streamlit >/dev/null 2>&1; then
  echo "Streamlit is not installed in the current environment." >&2
  echo "Install dependencies with: pip install -r requirements.txt" >&2
  exit 1
fi

echo "Starting Streamlit application..."
exec streamlit run app.py --server.headless true

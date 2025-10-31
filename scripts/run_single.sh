#!/usr/bin/env bash
# Wrapper for invoking the official SWE-bench harness.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <instance_id> <patch_path>" >&2
  exit 2
fi

INSTANCE_ID="$1"
PATCH_PATH="$2"

SWE_BENCH_ROOT=${SWE_BENCH_ROOT:-$HOME/SWE-bench}
HARNESS_MODULE=${HARNESS_MODULE:-swebench.harness.run_single}
CONFIG_PATH=${SWE_BENCH_CONFIG:-$SWE_BENCH_ROOT/configs/default.yaml}

if [ ! -d "$SWE_BENCH_ROOT" ]; then
  echo "SWE_BENCH_ROOT directory not found: $SWE_BENCH_ROOT" >&2
  exit 3
fi

if [ ! -f "$PATCH_PATH" ]; then
  echo "Patch file not found: $PATCH_PATH" >&2
  exit 4
fi

PYTHON_BIN=${PYTHON_BIN:-python}

exec "$PYTHON_BIN" scripts/run_single.py \
  --instance-id "$INSTANCE_ID" \
  --patch-path "$PATCH_PATH"

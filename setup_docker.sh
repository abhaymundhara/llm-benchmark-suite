#!/usr/bin/env bash
# Validate Docker availability for SWE-bench evaluation.

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH." >&2
  exit 1
fi

echo "Docker version:"
docker --version

echo "Checking Docker daemon availability..."
if ! docker info >/dev/null 2>&1; then
  echo "Unable to communicate with the Docker daemon. Is it running?" >&2
  exit 1
fi

echo "Docker daemon is reachable."

if [ -n "${SWE_BENCH_HARNESS_CMD:-}" ]; then
  echo "SWE_BENCH_HARNESS_CMD is configured: $SWE_BENCH_HARNESS_CMD"
else
  cat <<'MSG'
Warning: SWE_BENCH_HARNESS_CMD is not set.
Set this variable to the command that invokes the official SWE-bench harness.
Example: export SWE_BENCH_HARNESS_CMD="bash scripts/run_single.sh {instance_id} {patch_path}"
MSG
fi

echo "Docker environment check completed successfully."

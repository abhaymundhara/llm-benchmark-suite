#!/usr/bin/env python3
"""Run the SWE-bench harness for a single instance using a candidate patch."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("swebench.single")


def _env_bool(var_name: str, default: bool = False) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True, help="SWE-bench instance identifier.")
    parser.add_argument(
        "--patch-path",
        required=True,
        type=Path,
        help="Path to the candidate patch file produced by the model.",
    )

    parser.add_argument(
        "--dataset",
        default=os.getenv("SWE_BENCH_DATASET", "princeton-nlp/SWE-bench"),
        help="Dataset identifier (default: %(default)s or SWE_BENCH_DATASET env var).",
    )
    parser.add_argument(
        "--split",
        default=os.getenv("SWE_BENCH_SPLIT", "dev"),
        help="Dataset split (default: %(default)s or SWE_BENCH_SPLIT env var).",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("SWE_BENCH_MODEL_NAME", "llm-benchmark"),
        help="Model name reported to the harness.",
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("SWE_BENCH_RUN_ID"),
        help="Run identifier. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("SWE_BENCH_TIMEOUT", "1800")),
        help="Timeout (seconds) for harness execution.",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("SWE_BENCH_NAMESPACE"),
        help="Docker namespace (optional).",
    )
    parser.add_argument(
        "--base-image-tag",
        default=os.getenv("SWE_BENCH_BASE_IMAGE_TAG", "latest"),
        help="Base image tag for SWE-bench harness.",
    )
    parser.add_argument(
        "--env-image-tag",
        default=os.getenv("SWE_BENCH_ENV_IMAGE_TAG", "latest"),
        help="Environment image tag for SWE-bench harness.",
    )
    parser.add_argument(
        "--instance-image-tag",
        default=os.getenv("SWE_BENCH_INSTANCE_IMAGE_TAG", "latest"),
        help="Instance image tag for SWE-bench harness.",
    )
    parser.add_argument(
        "--arch",
        default=os.getenv("SWE_BENCH_ARCH", "x86_64"),
        help="Target architecture for harness images.",
    )
    parser.add_argument(
        "--keep-image",
        action="store_true",
        default=_env_bool("SWE_BENCH_KEEP_IMAGE", False),
        help="Do not remove Docker image after evaluation.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=_env_bool("SWE_BENCH_FORCE_REBUILD", False),
        help="Force rebuild of Docker images.",
    )
    parser.add_argument(
        "--rewrite-reports",
        action="store_true",
        default=_env_bool("SWE_BENCH_REWRITE_REPORTS", False),
        help="Rewrite existing reports if present.",
    )
    parser.add_argument(
        "--report-dir",
        default=os.getenv("SWE_BENCH_REPORT_DIR", "swebench_reports"),
        help="Directory for harness reports.",
    )
    parser.add_argument(
        "--cache-level",
        default=os.getenv("SWE_BENCH_CACHE_LEVEL", "dataset"),
        help="Cache level to use (dataset|project|none).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=_env_bool("SWE_BENCH_CLEAN", False),
        help="Clean Docker images after run.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def ensure_swebench_root() -> Path:
    root = os.getenv("SWE_BENCH_ROOT")
    if not root:
        raise EnvironmentError("SWE_BENCH_ROOT environment variable is not set.")
    path = Path(root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SWE_BENCH_ROOT directory not found: {path}")
    sys.path.insert(0, str(path))
    return path


def run_single_instance(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_swebench_root()

    from swebench.harness.run_evaluation import (
        load_swebench_dataset,
        make_test_spec,
        run_instance,
    )
    import docker

    patch_text = args.patch_path.read_text(encoding="utf-8")
    dataset_instances = load_swebench_dataset(args.dataset, args.split, [args.instance_id])
    if not dataset_instances:
        raise ValueError(
            f"Instance '{args.instance_id}' not found in dataset {args.dataset} ({args.split})."
        )

    instance = dataset_instances[0]
    test_spec = make_test_spec(
        instance,
        namespace=args.namespace,
        base_image_tag=args.base_image_tag,
        env_image_tag=args.env_image_tag,
        instance_image_tag=args.instance_image_tag,
        arch=args.arch,
    )

    client = docker.from_env()
    prediction = {
        "instance_id": args.instance_id,
        "model_name_or_path": args.model_name,
        "model_patch": patch_text,
    }
    run_id = args.run_id or f"single-{int(time.time())}"

    logger.info(
        "Running SWE-bench harness instance=%s dataset=%s split=%s run_id=%s",
        args.instance_id,
        args.dataset,
        args.split,
        run_id,
    )
    result = run_instance(
        test_spec=test_spec,
        pred=prediction,
        rm_image=not args.keep_image,
        force_rebuild=args.force_rebuild,
        client=client,
        run_id=run_id,
        timeout=args.timeout,
        rewrite_reports=args.rewrite_reports,
    )

    result_summary: Dict[str, Any] = {
        "instance_id": args.instance_id,
        "run_id": run_id,
        "status": result.get("status"),
        "exit_code": result.get("exit_code"),
        "passed": result.get("result") == "PASS",
        "report_path": result.get("report_path"),
        "stdout_path": result.get("stdout_path"),
        "stderr_path": result.get("stderr_path"),
    }
    logger.info("Harness completed with status=%s", result_summary["status"])

    report_dir = Path(args.report_dir).expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"{run_id}_{args.instance_id}_summary.json"
    summary_path.write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)

    return result_summary


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.patch_path.is_file():
        raise FileNotFoundError(f"Patch file not found: {args.patch_path}")
    result = run_single_instance(args)
    print(json.dumps(result, indent=2))
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()

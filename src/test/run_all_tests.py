#!/usr/bin/env python3
"""Run the maintained unit-test suite under src/test."""

import argparse
import subprocess
import sys
from pathlib import Path


TEST_SCRIPTS = [
    "test_config.py",
    "test_llm_config.py",
    "test_cache_manager.py",
    "test_db_connection.py",
    "test_sql_executor.py",
    "test_subplan.py",
    "test_knob_space.py",
    "test_result_exporter.py",
    "test_optimization_pipeline.py",
    "test_run_file_collection.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run maintained OBELISK test scripts")
    parser.add_argument(
        "--include-db-tests",
        action="store_true",
        help="Also run DB-dependent runtime validation scripts",
    )
    parser.add_argument(
        "--include-optimizer-tests",
        action="store_true",
        help="Also run optimizer-stack tests that may require torch/botorch runtime support",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_dir = Path(__file__).resolve().parent

    scripts = list(TEST_SCRIPTS)
    if args.include_optimizer_tests:
        scripts.extend(
            [
                "test_context_retrieval.py",
                "test_vector_workflow.py",
            ]
        )
    if args.include_db_tests:
        scripts.extend(
            [
                "test_best_config_runtime.py",
                "batch_best_config_runtime.py",
            ]
        )

    failed = []
    for script in scripts:
        script_path = test_dir / script
        if not script_path.exists():
            failed.append((script, "missing"))
            continue

        print(f"[RUN] {script}", flush=True)
        result = subprocess.run([sys.executable, str(script_path)], cwd=test_dir.parent.parent)
        if result.returncode != 0:
            failed.append((script, f"exit={result.returncode}"))

    if failed:
        print("\n[FAILED]")
        for name, reason in failed:
            print(f"- {name}: {reason}")
        return 1

    print("\n[OK] All selected tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure imports work when running from project root: `python src/test/test_best_config_runtime.py`
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.db_connection import DBConnectionFactory
from db.sql_executor import SQLExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Run 1 SQL with best_config hints vs default and report times")
    parser.add_argument("--result_json", required=True, help="Path to an OBELISK summary JSON containing best_config")
    parser.add_argument("--sql_file", required=True, help="Path to SQL file to execute")
    parser.add_argument("--db", default=None, help="Database name override (e.g., imdb). If not set, uses env/default")
    parser.add_argument("--port", type=int, default=4000, help="TiDB port (default: 4000)")
    parser.add_argument("--timeout_ms", type=int, default=180000, help="Max execution time in ms for EXPLAIN ANALYZE")
    parser.add_argument("--out_dir", default=None, help="Directory to save explain plans. Default: results/plan-dumps/<db>")
    return parser.parse_args()


def load_best_config(result_json_path: str) -> dict:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    best_config = data.get("best_config", {})
    if not isinstance(best_config, dict):
        raise ValueError("best_config not found or invalid in result json")
    return best_config


def run_twice_get_min(executor: SQLExecutor, sql_file: str, knobs: dict, timeout_ms: int):
    times = []
    explain_results = []
    for i in range(2):
        explain_result, _ = executor._get_explain_result(sql_file, knobs, timeout_ms)
        time_ms = executor._extract_time_seconds(explain_result[0])
        times.append(time_ms)
        explain_results.append(explain_result)
    # choose min time (handle None as +inf)
    comparable = [t if t is not None else float("inf") for t in times]
    min_idx = 0 if comparable[0] <= comparable[1] else 1
    min_time = times[min_idx] if times[min_idx] is not None else float("nan")
    min_explain = explain_results[min_idx]
    return min_time, times, min_explain


def main():
    args = parse_args()

    if args.db:
        # Ensure DB override takes effect for DBConnectionFactory(Config reads env on init)
        os.environ["TIDB_DB_NAME"] = args.db
    
    # Set port override
    os.environ["TIDB_PORT"] = str(args.port)

    connection = DBConnectionFactory.create_connection(autocommit=True)
    try:
        with connection.cursor() as cursor:
            executor = SQLExecutor(cursor, cache_name=args.db or "default")

            sql_stem = Path(args.sql_file).stem
            db_name = os.getenv('TIDB_DB_NAME', args.db or "default")
            out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "plan-dumps" / db_name
            default_plan_path = out_dir / f"{sql_stem}_default_explain.json"
            hinted_plan_path = out_dir / f"{sql_stem}_hinted_explain.json"

            # Default (no hints) - run twice and take min
            default_ms, default_runs, default_explain = run_twice_get_min(executor, args.sql_file, {}, args.timeout_ms)
            default_plan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(default_plan_path, "w", encoding="utf-8") as f:
                json.dump(default_explain, f, indent=2, ensure_ascii=False)

            # With best_config hints
            best_config = load_best_config(args.result_json)
            hinted_ms, hinted_runs, hinted_explain = run_twice_get_min(executor, args.sql_file, best_config, args.timeout_ms)
            hinted_plan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hinted_plan_path, "w", encoding="utf-8") as f:
                json.dump(hinted_explain, f, indent=2, ensure_ascii=False)

            speedup = None
            if default_ms and hinted_ms and default_ms > 0 and hinted_ms > 0:
                speedup = default_ms / hinted_ms

            print("==== Single SQL Runtime Test ====")
            print(f"SQL file: {args.sql_file}")
            print(f"DB: {db_name}")
            print(f"Timeout(ms): {args.timeout_ms}")
            print(f"Default runs (ms): {default_runs}")
            print(f"Default min  (ms): {default_ms}")
            print(f"Hinted runs  (ms): {hinted_runs}")
            print(f"Hinted min   (ms): {hinted_ms}")
            if speedup is not None:
                print(f"Speedup: {speedup:.2f}x")
            print(f"Saved default plan to: {default_plan_path}")
            print(f"Saved hinted  plan to: {hinted_plan_path}")
    finally:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

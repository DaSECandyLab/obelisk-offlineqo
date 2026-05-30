# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Ensure imports work when running from project root
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.db_connection import DBConnectionFactory
from db.sql_executor import SQLExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run best_config hints vs default for all summaries in a dir")
    parser.add_argument("--summary_dir", required=True, help="Directory containing OBELISK summary JSON files")
    parser.add_argument("--sql_base_dir", required=True, help="Base directory of SQL files (e.g., sql/job)")
    parser.add_argument("--db", default=None, help="Database name override (e.g., imdb)")
    parser.add_argument("--port", type=int, default=4000, help="TiDB port (default: 4000)")
    parser.add_argument("--timeout_ms", type=int, default=180000, help="Max execution time in ms for EXPLAIN ANALYZE")
    parser.add_argument("--out_dir", default=None, help="Directory to save explain plans and results CSV. Default: results/plan-dumps/<db>")
    parser.add_argument("--decimals", type=int, default=6, help="Number of decimals to keep in outputs")
    return parser.parse_args()


def load_best_config(result_json_path: Path) -> dict:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    best_config = data.get("best_config", {})
    sql_file = data.get("sql_file")
    if not isinstance(best_config, dict):
        raise ValueError(f"best_config invalid in {result_json_path}")
    return best_config, sql_file


def run_once(executor: SQLExecutor, sql_file: str, knobs: dict, timeout_ms: int):
    explain_result, _ = executor._get_explain_result(sql_file, knobs, timeout_ms)
    time_ms = executor._extract_time_seconds(explain_result[0])
    return time_ms, [time_ms], explain_result


def format_float(value: float, decimals: int) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "NaN"


def main():
    args = parse_args()

    if args.db:
        os.environ["TIDB_DB_NAME"] = args.db
    
    # Set port override
    os.environ["TIDB_PORT"] = str(args.port)

    db_name = os.getenv("TIDB_DB_NAME", args.db or "default")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "plan-dumps" / db_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "batch_runtime_results.csv"

    summary_dir = Path(args.summary_dir)
    sql_base_dir = Path(args.sql_base_dir)
    summary_files = sorted(summary_dir.glob("result_*_obelisk_summary.json"))
    if not summary_files:
        print(f"No summary JSON found in {summary_dir}")
        return

    connection = DBConnectionFactory.create_connection(autocommit=True)
    try:
        with connection.cursor() as cursor, open(results_csv, "w", newline="", encoding="utf-8") as csvfile:
            executor = SQLExecutor(cursor, cache_name=db_name)
            writer = csv.writer(csvfile)
            writer.writerow(["sql", "default_ms", "hinted_ms", "improvement_percent", "speedup"])

            for summary_path in summary_files:
                try:
                    best_config, sql_file_name = load_best_config(summary_path)
                    if not sql_file_name:
                        # fallback: infer from summary filename
                        sql_file_name = Path(summary_path).stem.replace("result_", "")
                        if not sql_file_name.endswith(".sql"):
                            sql_file_name += ".sql"

                    sql_path = sql_base_dir / sql_file_name
                    if not sql_path.exists():
                        print(f"[WARN] SQL file not found: {sql_path}, skip {summary_path.name}")
                        continue

                    stem = Path(sql_file_name).stem
                    default_plan_path = out_dir / f"{stem}_default_explain.json"
                    hinted_plan_path = out_dir / f"{stem}_hinted_explain.json"

                    print(f"=== Running {stem} (default) ===", flush=True)
                    default_ms, default_runs, default_explain = run_once(executor, str(sql_path), {}, args.timeout_ms)
                    print(f"{stem} default time: {format_float(default_ms, args.decimals)} ms", flush=True)
                    with open(default_plan_path, "w", encoding="utf-8") as f:
                        json.dump(default_explain, f, indent=2, ensure_ascii=False)

                    print(f"=== Running {stem} (hinted) ===", flush=True)
                    hinted_ms, hinted_runs, hinted_explain = run_once(executor, str(sql_path), best_config, args.timeout_ms)
                    print(f"{stem} hinted time: {format_float(hinted_ms, args.decimals)} ms", flush=True)
                    with open(hinted_plan_path, "w", encoding="utf-8") as f:
                        json.dump(hinted_explain, f, indent=2, ensure_ascii=False)

                    improvement = None
                    speedup = None
                    if default_ms and hinted_ms and default_ms > 0 and hinted_ms > 0:
                        improvement = (default_ms - hinted_ms) / default_ms * 100.0
                        speedup = default_ms / hinted_ms

                    print(f"[{stem}] default_time={format_float(default_ms, args.decimals)} hinted_time={format_float(hinted_ms, args.decimals)} "
                          f"impr={format_float(improvement if improvement is not None else float('nan'), args.decimals)}% "
                          f"speedup={format_float(speedup if speedup is not None else float('nan'), args.decimals)}x")

                    writer.writerow([
                        stem,
                        format_float(default_ms, args.decimals),
                        format_float(hinted_ms, args.decimals),
                        format_float(improvement if improvement is not None else float('nan'), args.decimals),
                        format_float(speedup if speedup is not None else float('nan'), args.decimals),
                    ])
                except Exception as e:
                    print(f"[ERROR] Failed on {summary_path.name}: {e}")
                    continue

        print(f"Saved CSV: {results_csv}")
    finally:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

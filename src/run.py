import argparse
import re
from pathlib import Path

from optimization.optimization_pipeline import SQLService
from tqdm import tqdm

from util.config import load_app_config
from util.logger import logger


RESULT_FILE_SUFFIX = "_obelisk.json"


def derive_repository_name(schema_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", schema_name or "").strip("._")
    return safe_name or "default"


def load_ceb_task_filter(sql_dir: Path) -> list[Path]:
    """
    When using the ceb-3k dataset, limit execution to queries listed in the
    .txt task files under the dataset root. Entries can be stems, filenames,
    or relative paths; queries may live in nested subdirectories.
    """
    task_files = sorted(sql_dir.glob("*.txt"))
    if not task_files:
        logger.warning(f"ceb-3k mode enabled but no .txt task files found under {sql_dir}")
        return []

    allowed_raw: set[str] = set()
    allowed_stems: set[str] = set()

    def add_entry(entry: str) -> None:
        normalized = entry.strip()
        if not normalized:
            return
        normalized = normalized.replace("\\", "/")

        lower_text = normalized.lower()
        allowed_raw.add(lower_text)
        allowed_raw.add(Path(normalized).name.lower())
        allowed_stems.add(Path(normalized).stem.lower())

        match = re.match(r"^ceb_(\d+)([a-zA-Z])(\d+)$", lower_text)
        if match:
            digits_prefix, letter, digits_suffix = match.groups()
            folder = f"{digits_prefix}{letter}"
            filename = f"{folder}{digits_suffix}.sql"
            allowed_raw.add(f"{folder}/{filename}")
            allowed_raw.add(filename)
            allowed_stems.add(Path(filename).stem)

    for task_file in task_files:
        try:
            with open(task_file, "r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh.readlines() if line.strip()]
            for line in lines:
                add_entry(line)
        except Exception:
            logger.exception(f"Failed to read task file {task_file}")

    all_sql = sorted(sql_dir.rglob("*.sql"))
    filtered = []
    for sql_file in all_sql:
        rel = sql_file.relative_to(sql_dir).as_posix().lower()
        if (
            rel in allowed_raw
            or sql_file.name.lower() in allowed_raw
            or sql_file.stem.lower() in allowed_stems
        ):
            filtered.append(sql_file)

    logger.info(
        f"ceb-3k task filter: {len(filtered)} of {len(all_sql)} SQL files matched task lists"
    )
    if not filtered:
        logger.warning("No SQL files matched ceb-3k task lists; verify task file contents")
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBELISK Optimizer")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config (default: etc/obelisk.toml)",
    )
    parser.add_argument(
        "--sql-dir",
        type=str,
        default=None,
        help="Directory containing SQL files; overrides config",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results; overrides config",
    )
    parser.add_argument(
        "--total-rounds",
        "--trials",
        dest="total_rounds",
        type=int,
        default=None,
        help="Total OBELISK rounds including Sobol warm-start rounds; overrides config",
    )
    parser.add_argument(
        "--warm-start-rounds",
        "--warm_times",
        "--warm-times",
        dest="warm_start_rounds",
        type=int,
        default=None,
        help="Sobol warm-start rounds; overrides config",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["vanilla_gp", "tcbo"],
        help="Optimization strategy; overrides config",
    )
    parser.add_argument(
        "--repository-name",
        type=str,
        default=None,
        help="Plan repository name under cache/<name>.db; overrides config",
    )
    parser.add_argument(
        "--cache-name",
        dest="repository_name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def collect_sql_files(sql_dir: Path) -> list[Path]:
    if sql_dir.name.lower() == "ceb-3k":
        sql_files = load_ceb_task_filter(sql_dir)
        if not sql_files:
            logger.warning("ceb-3k task filter produced no files; aborting")
            return []
        return sorted(sql_files)

    return sorted(sql_dir.rglob("*.sql"))


def build_result_path(results_dir: Path, sql_file: Path) -> Path:
    return results_dir / f"result_{sql_file.stem}{RESULT_FILE_SUFFIX}"


def individual_query_processing(
    service: SQLService,
    sql_file: Path,
    total_rounds: int,
    warm_start_rounds: int,
    results_dir: Path,
    strategy: str = "tcbo",
) -> None:
    results_file_path = build_result_path(results_dir, sql_file)
    logger.info(f"Start SQL file: {sql_file}")

    service.optimize(
        str(sql_file),
        str(results_file_path),
        total_rounds,
        warm_start_rounds,
        strategy,
    )
    logger.info(f"Done SQL file: {sql_file}")


def main() -> None:
    args = parse_args()
    app_config = load_app_config(args.config)
    run_config = app_config.run

    sql_dir = Path(args.sql_dir or run_config.sql_dir)
    if not sql_dir.exists():
        logger.error(f"SQL directory does not exist: {sql_dir}")
        return

    repository_name = (
        args.repository_name
        or run_config.repository_name
        or derive_repository_name(app_config.database.name)
    )
    results_dir_value = (
        args.results_dir
        if args.results_dir is not None
        else run_config.results_dir
    )
    results_dir = Path(results_dir_value) if results_dir_value else Path("results") / repository_name
    results_dir.mkdir(parents=True, exist_ok=True)
    sql_files = collect_sql_files(sql_dir)
    if not sql_files:
        logger.warning(f"No SQL files found in {sql_dir}")
        return

    total_rounds = (
        args.total_rounds if args.total_rounds is not None else run_config.total_rounds
    )
    warm_start_rounds = (
        args.warm_start_rounds
        if args.warm_start_rounds is not None
        else run_config.warm_start_rounds
    )
    strategy = args.strategy or run_config.strategy

    logger.info(f"OBELISK found {len(sql_files)} SQL queries to optimize.")

    failed_sql_files = []
    for sql_file in tqdm(sql_files, desc="Processing SQL files"):
        try:
            with SQLService(
                autocommit=app_config.database.autocommit,
                repository_name=repository_name,
                app_config=app_config,
            ) as service:
                individual_query_processing(
                    service,
                    sql_file,
                    total_rounds,
                    warm_start_rounds,
                    results_dir,
                    strategy,
                )
        except Exception:
            logger.exception(f"Failed to process {sql_file}")
            failed_sql_files.append(sql_file)

    if failed_sql_files:
        failed_names = ", ".join(str(path) for path in failed_sql_files[:5])
        if len(failed_sql_files) > 5:
            failed_names += ", ..."
        raise RuntimeError(
            f"Failed to optimize {len(failed_sql_files)} SQL file(s): {failed_names}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        raise

"""History-based conservative subplan latency estimates for Evaluator.

The estimator follows OBELISK Section 8.2: split historical latency into a
storage-layer component and a TiDB compute component, derive per-row costs from
the executed historical plan, then extrapolate them with optimizer row estimates
from the candidate plan.
"""

from __future__ import annotations

import re
from typing import Any

from util.logger import logger


PlanNode = dict[str, Any]

_TIME_UNITS_TO_MS = {
    "h": 60 * 60 * 1_000,
    "m": 60 * 1_000,
    "s": 1_000,
    "ms": 1,
    "us": 1 / 1_000,
    "µs": 1 / 1_000,
}


def parse_subplan_time(plan_details: PlanNode) -> tuple[float, float, float, float]:
    """Return (TiDB time, TiDB rows, TiKV time, TiKV rows) from a verbose plan."""
    tikv_subplans = get_tikv_subplans(plan_details)
    tikv_time = 0.0
    tikv_rows = 0.0

    for tikv_subplan in tikv_subplans:
        tikv_subplan_time = get_exec_time(tikv_subplan)
        tikv_subplan_rows = max(get_all_act_rows(tikv_subplan, True), default=0.0)
        if tikv_subplan_time > tikv_time:
            tikv_time = tikv_subplan_time
            tikv_rows = tikv_subplan_rows

    total_time = get_exec_time(plan_details)
    tidb_time = total_time - tikv_time
    tidb_rows = sum(get_all_act_rows(plan_details, False))
    return max(0.0, tidb_time), tidb_rows, tikv_time, tikv_rows


def parse_subplan_for_estimation(plan_details: PlanNode) -> tuple[float, float]:
    """Return candidate estimated rows as (TiDB rows, TiKV rows)."""
    tikv_est_rows = get_all_est_rows(plan_details, True)
    tikv_rows = max(tikv_est_rows) if tikv_est_rows else 0.0
    tidb_rows = sum(get_all_est_rows(plan_details, False))
    return tidb_rows, tikv_rows


def estimate_subplan_execution_time(
    est_plan: PlanNode,
    cached_plan: PlanNode,
) -> float | None:
    """Estimate candidate latency E(p, s) from one historical matched subplan."""
    try:
        tidb_time, tidb_rows, tikv_time, tikv_rows = parse_subplan_time(cached_plan)
        est_tidb_rows, est_tikv_rows = parse_subplan_for_estimation(est_plan)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Cannot estimate subplan latency from incomplete plan metrics: %s", exc)
        return None

    est_tidb_time = tidb_time * est_tidb_rows / tidb_rows if tidb_rows > 0 else 0.0
    est_tikv_time = tikv_time * est_tikv_rows / tikv_rows if tikv_rows > 0 else 0.0
    estimate = est_tidb_time + est_tikv_time

    logger.debug(
        "Subplan estimate: tidb rows=%s time=%.3fms, tikv rows=%s time=%.3fms, "
        "candidate tidb rows=%s tikv rows=%s estimate=%.3fms",
        tidb_rows,
        tidb_time,
        tikv_rows,
        tikv_time,
        est_tidb_rows,
        est_tikv_rows,
        estimate,
    )
    return estimate


def get_tikv_subplans(plan_node: PlanNode) -> list[PlanNode]:
    """Return storage-side subplans whose wall time contributes in parallel."""
    if is_tikv_task(plan_node) or is_tidb_tikv_connect_node(plan_node):
        return [plan_node]

    result: list[PlanNode] = []
    for sub_node in _suboperators(plan_node):
        result.extend(get_tikv_subplans(sub_node))
    return result


def is_tidb_tikv_connect_node(plan_node: PlanNode) -> bool:
    """Return whether a TiDB root node directly wraps TiKV work."""
    return _task_type(plan_node) == "root" and any(
        is_tikv_task(sub_node)
        for sub_node in _suboperators(plan_node)
    )


def is_tikv_task(plan_node: PlanNode) -> bool:
    return _task_type(plan_node) == "cop[tikv]"


def is_leaf(plan_node: PlanNode) -> bool:
    return not _suboperators(plan_node)


def get_all_act_rows(plan_node: PlanNode, for_tikv: bool) -> list[float]:
    rows: list[float] = []
    if for_tikv and (is_tikv_task(plan_node) or is_tidb_tikv_connect_node(plan_node)):
        row_count = _float_field(plan_node, "actRows")
        if row_count is not None:
            rows.append(row_count)
    elif not for_tikv and _task_type(plan_node) == "root" and not is_tidb_tikv_connect_node(plan_node):
        row_count = _float_field(plan_node, "actRows")
        if row_count is not None:
            rows.append(row_count)

    for sub_node in _suboperators(plan_node):
        rows.extend(get_all_act_rows(sub_node, for_tikv))
    return rows


def get_exec_time(plan_node: PlanNode) -> float:
    execute_info = str(plan_node.get("executeInfo", ""))
    for pattern in (
        r"(?<![a-zA-Z_])time:\s*(\d+(?:\.\d+)?)(h|ms|us|µs|s|m)",
        r"total_time:\s*(\d+(?:\.\d+)?)(h|ms|us|µs|s|m)",
        r"proc max:\s*(\d+(?:\.\d+)?)(h|ms|us|µs|s|m)",
    ):
        match = re.search(pattern, execute_info)
        if match:
            value, unit = match.groups()
            return float(value) * _TIME_UNITS_TO_MS[unit]
    raise ValueError(f"No supported execution time found in: {execute_info}")


def get_all_est_rows(plan_node: PlanNode, for_tikv: bool) -> list[float]:
    rows: list[float] = []
    if for_tikv and (is_tikv_task(plan_node) or is_tidb_tikv_connect_node(plan_node)):
        row_count = _float_field(plan_node, "estRows")
        if row_count is not None:
            rows.append(row_count)
    elif not for_tikv and _task_type(plan_node) == "root" and not is_tidb_tikv_connect_node(plan_node):
        row_count = _float_field(plan_node, "estRows")
        if row_count is not None:
            rows.append(row_count)

    for sub_node in _suboperators(plan_node):
        rows.extend(get_all_est_rows(sub_node, for_tikv))
    return rows


def _suboperators(plan_node: PlanNode) -> list[PlanNode]:
    suboperators = plan_node.get("subOperators", [])
    if not isinstance(suboperators, list):
        return []
    return [sub_node for sub_node in suboperators if isinstance(sub_node, dict)]


def _task_type(plan_node: PlanNode) -> str:
    return str(plan_node.get("taskType", ""))


def _float_field(plan_node: PlanNode, field: str) -> float | None:
    try:
        return float(plan_node[field])
    except (KeyError, TypeError, ValueError):
        return None

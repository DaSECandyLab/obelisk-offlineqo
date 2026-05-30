"""OBELISK Evaluator status definitions."""

from enum import Enum


class EvaluationStatus(str, Enum):
    """Canonical outcomes from the two-stage performance Evaluator."""

    EXECUTED = "executed"
    TIMEOUT = "timeout"
    DUPLICATE_PLAN = "duplicate_plan"
    SUBPLAN_REJECTED = "subplan_rejected"
    ADMITTED = "admitted"


def is_admission_estimate(status: EvaluationStatus | str) -> bool:
    return status == EvaluationStatus.SUBPLAN_REJECTED or status == EvaluationStatus.SUBPLAN_REJECTED.value


def is_true_observation(status: EvaluationStatus | str) -> bool:
    """Return whether status is not an admission estimate.

    Timeout censoring requires the configured tau and is handled by the
    optimization pipeline where execution time and tau are both available.
    """
    return not (
        is_admission_estimate(status)
        or status == EvaluationStatus.TIMEOUT
        or status == EvaluationStatus.TIMEOUT.value
    )

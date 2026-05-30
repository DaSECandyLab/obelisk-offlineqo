"""Optimization package with lazy exports to avoid heavy import side effects."""

from importlib import import_module
from typing import Any

__all__ = [
    "SQLService",
    "OptimizationPipeline",
    "Guider",
    "BayesianGuider",
    "Tuner",
    "EvaluationStatus",
    "ResultExporter",
    "DatabaseService",
]

_EXPORT_MAP = {
    "SQLService": "optimization.optimization_pipeline",
    "OptimizationPipeline": "optimization.optimization_pipeline",
    "Guider": "optimization.guider",
    "BayesianGuider": "optimization.guider",
    "Tuner": "optimization.guider",
    "EvaluationStatus": "optimization.evaluator",
    "ResultExporter": "optimization.result_exporter",
    "DatabaseService": "optimization.database_service",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORT_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module 'optimization' has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value

"""
Event definitions for optimization pipeline
"""
from dataclasses import dataclass
from typing import Any, Dict, List

from util.logger import logger


@dataclass
class OptimizationEvent:
    """Base class for all optimization events"""
    timestamp: float
    event_type: str
    context: Dict[str, Any]


@dataclass
class OptimizationStarted(OptimizationEvent):
    """Event when optimization starts"""
    sql_filepath: str
    result_filepath: str
    total_rounds: int
    warm_start_rounds: int
    strategy: str

    @property
    def total_trials(self) -> int:
        """Backward-compatible name for older event consumers."""
        return self.total_rounds

    @property
    def warm_start_times(self) -> int:
        """Backward-compatible name for older event consumers."""
        return self.warm_start_rounds


@dataclass
class OptimizationCompleted(OptimizationEvent):
    """Event when optimization completes"""
    total_duration: float
    results: List[Dict[str, Any]]
    best_result: Dict[str, Any]


class EventDispatcher:
    """Simple event dispatcher to manage event subscriptions"""

    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_type: str, callback):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event: OptimizationEvent):
        """Publish an event to all subscribers"""
        event_type = event.__class__.__name__
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception:
                    logger.exception("Event subscriber failed for %s", event_type)
                    raise

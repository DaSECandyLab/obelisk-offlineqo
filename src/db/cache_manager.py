"""Backward-compatible alias for the OBELISK plan repository."""

from db.plan_repository import PlanRepository


CacheManager = PlanRepository

__all__ = ["CacheManager"]

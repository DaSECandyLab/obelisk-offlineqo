"""Backward-compatible exports for the OBELISK Configuration Reasoner."""

from llm.reasoner import ConfigurationReasoner

LLM = ConfigurationReasoner

__all__ = ["ConfigurationReasoner", "LLM"]

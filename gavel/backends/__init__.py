"""Backend implementations for gavel."""

from .openai_compat import BackendError, OpenAICompatBackend

__all__ = ["BackendError", "OpenAICompatBackend"]

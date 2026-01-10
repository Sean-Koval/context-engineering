"""Structured event logging for context operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field


class EventLevel(str, Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ContextEvent(BaseModel):
    """Structured context event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    level: EventLevel
    event_type: str
    session_id: str | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class ContextEvents:
    """
    Structured event logging for context operations.

    Uses structlog for machine-parseable logs with rich context.

    Example:
        >>> events = ContextEvents(service_name="my-agent")
        >>> events.session_started("session-123", total_tokens=100000)
        >>> events.compression_completed("session-123", "dedupe", 1000, 300, 50.5)
    """

    def __init__(
        self,
        service_name: str = "context-engine",
        log_level: str = "INFO",
        json_format: bool = True,
    ) -> None:
        """
        Initialize event logging.

        Args:
            service_name: Service name for log identification
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_format: If True, output JSON; otherwise, use console format
        """
        self._service_name = service_name
        self._json_format = json_format

        # Configure stdlib logging
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level.upper(), logging.INFO),
        )

        # Configure structlog processors
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        self._logger = structlog.get_logger(service_name)

    def _log(
        self,
        level: EventLevel,
        event_type: str,
        message: str,
        session_id: str | None = None,
        **data: Any,
    ) -> ContextEvent:
        """
        Internal logging method.

        Args:
            level: Event severity level
            event_type: Type of event
            message: Human-readable message
            session_id: Session identifier (optional)
            **data: Additional structured data

        Returns:
            The created ContextEvent
        """
        event = ContextEvent(
            level=level,
            event_type=event_type,
            session_id=session_id,
            message=message,
            data=data,
        )

        log_method = getattr(self._logger, level.value)
        log_method(
            message,
            event_type=event_type,
            session_id=session_id,
            **data,
        )

        return event

    # Session lifecycle events

    def session_started(
        self,
        session_id: str,
        total_tokens: int,
        **kwargs: Any,
    ) -> ContextEvent:
        """
        Log session start.

        Args:
            session_id: Session identifier
            total_tokens: Total token budget for the session
            **kwargs: Additional context data
        """
        return self._log(
            EventLevel.INFO,
            "session.started",
            f"Session started with {total_tokens:,} token budget",
            session_id=session_id,
            total_tokens=total_tokens,
            **kwargs,
        )

    def session_ended(
        self,
        session_id: str,
        duration_seconds: float,
        tokens_used: int,
        compressions: int,
        **kwargs: Any,
    ) -> ContextEvent:
        """
        Log session end.

        Args:
            session_id: Session identifier
            duration_seconds: Session duration
            tokens_used: Total tokens used
            compressions: Number of compression operations
            **kwargs: Additional context data
        """
        return self._log(
            EventLevel.INFO,
            "session.ended",
            f"Session ended after {duration_seconds:.1f}s, used {tokens_used:,} tokens",
            session_id=session_id,
            duration_seconds=duration_seconds,
            tokens_used=tokens_used,
            compressions=compressions,
            **kwargs,
        )

    # Budget events

    def pre_rot_warning(
        self,
        session_id: str,
        usage_ratio: float,
        tokens_used: int,
        tokens_total: int,
    ) -> ContextEvent:
        """
        Log pre-rot warning.

        Args:
            session_id: Session identifier
            usage_ratio: Current usage ratio (0.0-1.0)
            tokens_used: Tokens currently used
            tokens_total: Total token budget
        """
        return self._log(
            EventLevel.WARNING,
            "budget.pre_rot_warning",
            f"Pre-rot warning: {usage_ratio:.1%} token usage ({tokens_used:,}/{tokens_total:,})",
            session_id=session_id,
            usage_ratio=usage_ratio,
            tokens_used=tokens_used,
            tokens_total=tokens_total,
        )

    def pre_rot_trigger(
        self,
        session_id: str,
        usage_ratio: float,
        compression_initiated: bool,
    ) -> ContextEvent:
        """
        Log pre-rot trigger.

        Args:
            session_id: Session identifier
            usage_ratio: Current usage ratio (0.0-1.0)
            compression_initiated: Whether compression was started
        """
        status = "started" if compression_initiated else "pending"
        return self._log(
            EventLevel.WARNING,
            "budget.pre_rot_trigger",
            f"Pre-rot trigger at {usage_ratio:.1%}, compression={status}",
            session_id=session_id,
            usage_ratio=usage_ratio,
            compression_initiated=compression_initiated,
        )

    # Compression events

    def compression_started(
        self,
        session_id: str,
        strategy: str,
        target_tokens: int | None = None,
    ) -> ContextEvent:
        """
        Log compression start.

        Args:
            session_id: Session identifier
            strategy: Compression strategy name
            target_tokens: Target tokens to save (optional)
        """
        msg = f"Starting compression with {strategy}"
        if target_tokens:
            msg += f", targeting {target_tokens:,} tokens"
        return self._log(
            EventLevel.INFO,
            "compression.started",
            msg,
            session_id=session_id,
            strategy=strategy,
            target_tokens=target_tokens,
        )

    def compression_completed(
        self,
        session_id: str,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
        duration_ms: float,
    ) -> ContextEvent:
        """
        Log compression completion.

        Args:
            session_id: Session identifier
            strategy: Compression strategy name
            original_tokens: Tokens before compression
            compressed_tokens: Tokens after compression
            duration_ms: Duration in milliseconds
        """
        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else float("inf")
        saved = original_tokens - compressed_tokens

        return self._log(
            EventLevel.INFO,
            "compression.completed",
            f"Compression {strategy}: {ratio:.1f}x, saved {saved:,} tokens in {duration_ms:.1f}ms",
            session_id=session_id,
            strategy=strategy,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            tokens_saved=saved,
            duration_ms=duration_ms,
        )

    def compression_failed(
        self,
        session_id: str,
        strategy: str,
        error: str,
    ) -> ContextEvent:
        """
        Log compression failure.

        Args:
            session_id: Session identifier
            strategy: Compression strategy name
            error: Error message
        """
        return self._log(
            EventLevel.ERROR,
            "compression.failed",
            f"Compression {strategy} failed: {error}",
            session_id=session_id,
            strategy=strategy,
            error=error,
        )

    # Cache events

    def cache_hit(
        self,
        session_id: str,
        tool_name: str,
        tokens_saved: int,
    ) -> ContextEvent:
        """
        Log cache hit.

        Args:
            session_id: Session identifier
            tool_name: Name of the cached tool
            tokens_saved: Tokens saved by cache hit
        """
        return self._log(
            EventLevel.DEBUG,
            "cache.hit",
            f"Cache hit for {tool_name}, saved {tokens_saved:,} tokens",
            session_id=session_id,
            tool_name=tool_name,
            tokens_saved=tokens_saved,
        )

    def cache_miss(
        self,
        session_id: str,
        tool_name: str,
    ) -> ContextEvent:
        """
        Log cache miss.

        Args:
            session_id: Session identifier
            tool_name: Name of the tool
        """
        return self._log(
            EventLevel.DEBUG,
            "cache.miss",
            f"Cache miss for {tool_name}",
            session_id=session_id,
            tool_name=tool_name,
        )

    # Retrieval events

    def retrieval_completed(
        self,
        session_id: str,
        strategy: str,
        results_count: int,
        latency_ms: float,
    ) -> ContextEvent:
        """
        Log retrieval completion.

        Args:
            session_id: Session identifier
            strategy: Retrieval strategy name
            results_count: Number of results retrieved
            latency_ms: Latency in milliseconds
        """
        return self._log(
            EventLevel.DEBUG,
            "retrieval.completed",
            f"Retrieved {results_count} results via {strategy} in {latency_ms:.1f}ms",
            session_id=session_id,
            strategy=strategy,
            results_count=results_count,
            latency_ms=latency_ms,
        )

    # Graph events

    def graph_node_added(
        self,
        session_id: str,
        node_type: str,
        node_id: str,
        token_count: int | None = None,
    ) -> ContextEvent:
        """
        Log node addition to graph.

        Args:
            session_id: Session identifier
            node_type: Type of node added
            node_id: Node identifier
            token_count: Token count of node (optional)
        """
        return self._log(
            EventLevel.DEBUG,
            "graph.node_added",
            f"Added {node_type} node {node_id}",
            session_id=session_id,
            node_type=node_type,
            node_id=node_id,
            token_count=token_count,
        )

    def graph_node_removed(
        self,
        session_id: str,
        node_id: str,
        reason: str,
    ) -> ContextEvent:
        """
        Log node removal from graph.

        Args:
            session_id: Session identifier
            node_id: Node identifier
            reason: Reason for removal
        """
        return self._log(
            EventLevel.DEBUG,
            "graph.node_removed",
            f"Removed node {node_id}: {reason}",
            session_id=session_id,
            node_id=node_id,
            reason=reason,
        )

    # Error events

    def error(
        self,
        session_id: str | None,
        error_type: str,
        message: str,
        **kwargs: Any,
    ) -> ContextEvent:
        """
        Log error event.

        Args:
            session_id: Session identifier (optional)
            error_type: Type/category of error
            message: Error message
            **kwargs: Additional error context
        """
        return self._log(
            EventLevel.ERROR,
            f"error.{error_type}",
            message,
            session_id=session_id,
            **kwargs,
        )

    def critical(
        self,
        session_id: str | None,
        error_type: str,
        message: str,
        **kwargs: Any,
    ) -> ContextEvent:
        """
        Log critical error event.

        Args:
            session_id: Session identifier (optional)
            error_type: Type/category of error
            message: Error message
            **kwargs: Additional error context
        """
        return self._log(
            EventLevel.CRITICAL,
            f"critical.{error_type}",
            message,
            session_id=session_id,
            **kwargs,
        )

    # Generic logging

    def info(
        self,
        event_type: str,
        message: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> ContextEvent:
        """Log an info event."""
        return self._log(EventLevel.INFO, event_type, message, session_id, **kwargs)

    def debug(
        self,
        event_type: str,
        message: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> ContextEvent:
        """Log a debug event."""
        return self._log(EventLevel.DEBUG, event_type, message, session_id, **kwargs)

    def warning(
        self,
        event_type: str,
        message: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> ContextEvent:
        """Log a warning event."""
        return self._log(EventLevel.WARNING, event_type, message, session_id, **kwargs)

"""Configuration for observability components."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ObservabilityConfig(BaseModel):
    """Configuration for the observability stack."""

    # Service identification
    service_name: str = Field(default="context-engine", description="Service name for tracing")
    service_version: str = Field(default="0.1.0", description="Service version")
    environment: str = Field(default="development", description="Deployment environment")

    # Tracing config
    tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    trace_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate (0.0-1.0)"
    )

    # Metrics config
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_export_interval_ms: int = Field(
        default=10000, ge=1000, description="Metrics export interval in milliseconds"
    )

    # Events/logging config
    events_enabled: bool = Field(default=True, description="Enable structured event logging")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level"
    )
    log_format: Literal["json", "console"] = Field(default="json", description="Log output format")

    # Exporter config
    exporter_type: Literal["console", "otlp", "prometheus", "none"] = Field(
        default="console", description="Primary exporter type"
    )
    otlp_endpoint: str | None = Field(
        default=None, description="OTLP collector endpoint (e.g., http://localhost:4317)"
    )
    prometheus_port: int = Field(
        default=9090, ge=1024, le=65535, description="Prometheus metrics port"
    )

    # Alert thresholds
    pre_rot_warning_threshold: float = Field(
        default=0.50, ge=0.0, le=1.0, description="Token usage ratio for pre-rot warning"
    )
    pre_rot_trigger_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Token usage ratio for pre-rot trigger"
    )
    cache_hit_rate_warning: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Cache hit rate below this triggers warning"
    )
    retrieval_latency_warning_ms: float = Field(
        default=200.0, ge=0.0, description="Retrieval latency above this triggers warning"
    )

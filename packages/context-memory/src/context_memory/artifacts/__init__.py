"""Artifact management for context-memory.

Provides versioned artifact storage with content addressing:
- ArtifactManager: Main management class
- Artifact: Versioned artifact model
- ArtifactVersion: Individual version metadata
- DiffResult: Version comparison result

Example:
    >>> from context_memory.artifacts import ArtifactManager
    >>> manager = ArtifactManager()
    >>> artifact = await manager.create_artifact(
    ...     session_id="sess-123",
    ...     name="config.json",
    ...     content='{"key": "value"}',
    ... )
"""

from context_memory.artifacts.manager import (
    Artifact,
    ArtifactManager,
    ArtifactVersion,
    DiffResult,
)

__all__ = [
    "Artifact",
    "ArtifactManager",
    "ArtifactVersion",
    "DiffResult",
]

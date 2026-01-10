"""Type definitions for the context graph.

This module defines the core enums used throughout the context graph:
- NodeType: Types of nodes (message, tool_call, artifact, etc.)
- EdgeType: Types of relationships between nodes
- CompressionLevel: Compression state of a node
- Role: Message roles for MESSAGE nodes
"""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the context graph.

    Each node type represents a different kind of context item:
    - MESSAGE: User or assistant conversation messages
    - TOOL_CALL: Tool/function invocations
    - TOOL_RESULT: Responses from tool executions
    - ARTIFACT: Code, files, or structured data
    - ENTITY: Extracted entity references (people, files, concepts)
    - SUMMARY: Compressed representations of other nodes
    - SYSTEM: System prompts and instructions
    - MEMORY: Items retrieved from long-term storage
    """

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ARTIFACT = "artifact"
    ENTITY = "entity"
    SUMMARY = "summary"
    SYSTEM = "system"
    MEMORY = "memory"


class EdgeType(str, Enum):
    """Types of relationships between nodes.

    Edges capture semantic relationships in the context graph:
    - TEMPORAL: Sequential ordering (previous -> next)
    - CAUSAL: Causation relationship (A caused B)
    - REFERENCES: A mentions or uses B
    - SUMMARIZES: Summary node points to original nodes
    - CONTRADICTS: A contradicts information in B
    - DEPENDS_ON: A requires B to be meaningful
    - SAME_ENTITY: Coreference link (same entity, different mentions)
    - PARENT_CHILD: Hierarchical relationship
    - TOOL_IO: Links tool_call to its tool_result
    """

    TEMPORAL = "temporal"
    CAUSAL = "causal"
    REFERENCES = "references"
    SUMMARIZES = "summarizes"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    SAME_ENTITY = "same_entity"
    PARENT_CHILD = "parent_child"
    TOOL_IO = "tool_io"


class CompressionLevel(int, Enum):
    """Compression state of a node.

    Represents the level of compression applied to a node's content:
    - FULL (0): Original content, no compression applied
    - COMPACTED (1): Reversible compression (externalized, deduplicated)
    - SUMMARIZED (2): Irreversible summarization applied
    - EVICTED (3): Content externalized, only pointer remains

    Higher values indicate more aggressive compression.
    Compression should proceed FULL -> COMPACTED -> SUMMARIZED -> EVICTED.
    """

    FULL = 0
    COMPACTED = 1
    SUMMARIZED = 2
    EVICTED = 3


class Role(str, Enum):
    """Message roles for MESSAGE nodes.

    Standard roles used in LLM conversations:
    - USER: Messages from the human user
    - ASSISTANT: Messages from the AI assistant
    - SYSTEM: System prompts and instructions
    - TOOL: Tool/function responses (used by some LLM APIs)
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

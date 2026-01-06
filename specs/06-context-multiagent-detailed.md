# Context-Multiagent: Detailed Implementation Specification

> **Phase 4 Multi-Agent** | Estimated: 4-5 weeks | Priority: P2
> This document provides implementable specifications for multi-agent context coordination.

---

## Table of Contents
1. [Package Overview](#package-overview)
2. [Component 1: ContextBroker](#component-1-contextbroker)
3. [Component 2: ContextHandoff](#component-2-contexthandoff)
4. [Component 3: SharedMemoryPool](#component-3-sharedmemorypool)
5. [Component 4: ContextSync](#component-4-contextsync)
6. [Integration Patterns](#integration-patterns)
7. [Task Breakdown](#task-breakdown)

---

## Package Overview

### Purpose
`context-multiagent` enables context sharing between multiple LLM agents, supporting handoffs, collaborative memory, and real-time synchronization.

### Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT CONTEXT FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐           │
│  │   Agent A   │       │   Agent B   │       │   Agent C   │           │
│  │  (Planner)  │       │  (Coder)    │       │  (Reviewer) │           │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘           │
│         │                     │                     │                    │
│         │    Handoff          │    Sync             │                    │
│         │    ─────────►       │    ◄──────────►     │                    │
│         │                     │                     │                    │
│         └──────────┬──────────┴──────────┬──────────┘                   │
│                    │                     │                               │
│                    ▼                     ▼                               │
│            ┌─────────────────────────────────────┐                      │
│            │          ContextBroker              │                      │
│            │                                     │                      │
│            │  • Agent registry                   │                      │
│            │  • Sharing policies                 │                      │
│            │  • Subgraph routing                 │                      │
│            └─────────────────┬───────────────────┘                      │
│                              │                                           │
│                              ▼                                           │
│            ┌─────────────────────────────────────┐                      │
│            │        SharedMemoryPool             │                      │
│            │                                     │                      │
│            │  • Conflict resolution              │                      │
│            │  • Access control                   │                      │
│            │  • Subscriptions                    │                      │
│            └─────────────────────────────────────┘                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Subgraph sharing**: Agents share relevant subgraphs, not full context
2. **Briefing generation**: Handoffs include AI-generated task briefings
3. **Conflict resolution**: Last-writer-wins with optional merge strategies
4. **Access control**: Per-agent permissions for read/write/subscribe

### Dependencies

```toml
[project]
name = "context-multiagent"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "context-core>=0.1.0",
    "context-memory>=0.1.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
redis = ["redis>=5.0"]
kafka = ["aiokafka>=0.9"]
all = ["context-multiagent[redis]"]
```

### Module Structure

```
context_multiagent/
├── __init__.py
├── py.typed
├── types.py                  # Shared types
├── broker.py                 # ContextBroker
├── handoff.py                # ContextHandoff
├── shared.py                 # SharedMemoryPool
├── sync.py                   # ContextSync
├── policies/
│   ├── __init__.py
│   ├── sharing.py            # SharingPolicy
│   └── access.py             # AccessPolicy
└── transport/
    ├── __init__.py
    ├── base.py               # MessageTransport protocol
    ├── memory.py             # InMemoryTransport
    └── redis.py              # RedisTransport
```

---

## Component 1: ContextBroker

### 1.1 Type Definitions

```python
# context_multiagent/types.py
from enum import Enum
from typing import Any, Optional, Set
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

class AgentRole(str, Enum):
    """Common agent roles."""
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    ORCHESTRATOR = "orchestrator"
    CUSTOM = "custom"


class AgentState(str, Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"      # Waiting for handoff
    HANDOFF = "handoff"      # Performing handoff
    TERMINATED = "terminated"


class AgentInfo(BaseModel):
    """Information about a registered agent."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    role: AgentRole
    state: AgentState = AgentState.IDLE
    capabilities: Set[str] = Field(default_factory=set)

    # Session info
    session_id: Optional[str] = None
    current_task: Optional[str] = None

    # Stats
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    handoffs_sent: int = 0
    handoffs_received: int = 0


class SharingScope(str, Enum):
    """Scope of context sharing."""
    NONE = "none"            # No sharing
    TASK = "task"            # Share task-relevant context
    SESSION = "session"      # Share full session context
    GLOBAL = "global"        # Share across sessions


class SharedContext(BaseModel):
    """Context shared between agents."""
    id: UUID = Field(default_factory=uuid4)
    source_agent_id: UUID
    target_agent_ids: Set[UUID]

    # Content
    graph_data: dict           # Serialized ContextGraph
    briefing: Optional[str] = None
    task: Optional[str] = None

    # Metadata
    scope: SharingScope
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    token_count: int = 0


class ContextMessage(BaseModel):
    """Message for context synchronization."""
    id: UUID = Field(default_factory=uuid4)
    type: str  # "update", "handoff", "subscribe", "unsubscribe"
    sender_id: UUID
    recipient_ids: Optional[Set[UUID]] = None  # None = broadcast

    payload: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 1.2 Sharing Policies

```python
# context_multiagent/policies/sharing.py
from typing import Optional, Set, Callable
from uuid import UUID
from pydantic import BaseModel

from context_core.graph import ContextGraph, ContextNode, NodeType

from ..types import SharingScope, AgentRole


class SharingRule(BaseModel):
    """Rule for what context to share."""
    name: str
    priority: int = 0

    # Filters
    node_types: Optional[Set[str]] = None
    min_importance: Optional[float] = None
    max_age_seconds: Optional[int] = None
    required_tags: Optional[Set[str]] = None

    # Actions
    include: bool = True  # True = include, False = exclude


class SharingPolicy(BaseModel):
    """Policy for sharing context between agents."""
    name: str
    description: str = ""

    # Who can share
    source_roles: Optional[Set[str]] = None
    target_roles: Optional[Set[str]] = None

    # What to share
    scope: SharingScope = SharingScope.TASK
    rules: list[SharingRule] = []

    # Limits
    max_tokens: Optional[int] = None
    max_nodes: Optional[int] = None

    def filter_graph(
        self,
        graph: ContextGraph,
        source_agent: "AgentInfo",
        target_agent: "AgentInfo",
    ) -> ContextGraph:
        """Apply policy to filter graph for sharing."""
        # Check role permissions
        if self.source_roles and source_agent.role.value not in self.source_roles:
            return ContextGraph()  # Empty graph

        if self.target_roles and target_agent.role.value not in self.target_roles:
            return ContextGraph()

        # Start with all nodes
        included_ids = set()
        excluded_ids = set()

        # Apply rules in priority order
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for node in graph:
            for rule in sorted_rules:
                if self._matches_rule(node, rule):
                    if rule.include:
                        included_ids.add(node.id)
                    else:
                        excluded_ids.add(node.id)
                    break  # First matching rule wins

        # Build filtered graph
        final_ids = included_ids - excluded_ids

        # Apply limits
        if self.max_nodes and len(final_ids) > self.max_nodes:
            # Keep most important
            nodes = [graph.get_node(nid) for nid in final_ids]
            nodes.sort(key=lambda n: n.compute_importance(), reverse=True)
            final_ids = {n.id for n in nodes[:self.max_nodes]}

        return graph.get_subgraph(list(final_ids), max_depth=0)

    def _matches_rule(self, node: ContextNode, rule: SharingRule) -> bool:
        """Check if node matches rule criteria."""
        if rule.node_types and node.type.value not in rule.node_types:
            return False

        if rule.min_importance and node.compute_importance() < rule.min_importance:
            return False

        if rule.required_tags and not (rule.required_tags & node.metadata.tags):
            return False

        # Age check would need current time
        return True


# Default policies
DEFAULT_POLICIES = {
    "task_handoff": SharingPolicy(
        name="task_handoff",
        description="Share task-relevant context during handoffs",
        scope=SharingScope.TASK,
        rules=[
            SharingRule(name="include_messages", node_types={"message"}, priority=10, include=True),
            SharingRule(name="include_artifacts", node_types={"artifact"}, priority=10, include=True),
            SharingRule(name="include_important", min_importance=0.7, priority=5, include=True),
            SharingRule(name="exclude_tool_results", node_types={"tool_result"}, priority=3, include=False),
        ],
        max_tokens=50000,
    ),
    "reviewer_context": SharingPolicy(
        name="reviewer_context",
        description="Share code and discussion for review",
        scope=SharingScope.TASK,
        target_roles={"reviewer"},
        rules=[
            SharingRule(name="include_code", node_types={"artifact"}, required_tags={"code"}, include=True),
            SharingRule(name="include_discussion", node_types={"message"}, include=True),
        ],
    ),
}
```

### 1.3 ContextBroker Class

```python
# context_multiagent/broker.py
from typing import Optional, Dict, Set, Callable, Awaitable
from uuid import UUID
import asyncio

from context_core.graph import ContextGraph

from .types import (
    AgentInfo, AgentState, AgentRole, SharedContext,
    SharingScope, ContextMessage,
)
from .policies.sharing import SharingPolicy, DEFAULT_POLICIES
from .transport.base import MessageTransport


class ContextBroker:
    """
    Central coordinator for multi-agent context sharing.

    Features:
    - Agent registration and discovery
    - Policy-based context sharing
    - Subgraph routing
    - Message broadcasting
    """

    def __init__(
        self,
        transport: Optional[MessageTransport] = None,
        default_policy: Optional[SharingPolicy] = None,
    ):
        self._agents: Dict[UUID, AgentInfo] = {}
        self._policies: Dict[str, SharingPolicy] = dict(DEFAULT_POLICIES)
        self._default_policy = default_policy or self._policies.get("task_handoff")
        self._transport = transport

        # Subscriptions: agent_id -> set of topics
        self._subscriptions: Dict[UUID, Set[str]] = {}

        # Message handlers
        self._handlers: Dict[str, Callable] = {}

        # Shared contexts
        self._shared_contexts: Dict[UUID, SharedContext] = {}

    async def register_agent(
        self,
        name: str,
        role: AgentRole,
        capabilities: Optional[Set[str]] = None,
        session_id: Optional[str] = None,
    ) -> AgentInfo:
        """Register an agent with the broker."""
        agent = AgentInfo(
            name=name,
            role=role,
            capabilities=capabilities or set(),
            session_id=session_id,
        )

        self._agents[agent.id] = agent
        self._subscriptions[agent.id] = set()

        # Notify other agents
        if self._transport:
            await self._broadcast(ContextMessage(
                type="agent_registered",
                sender_id=agent.id,
                payload={"agent": agent.model_dump(mode="json")},
            ))

        return agent

    async def unregister_agent(self, agent_id: UUID) -> bool:
        """Unregister an agent."""
        if agent_id not in self._agents:
            return False

        agent = self._agents.pop(agent_id)
        self._subscriptions.pop(agent_id, None)

        if self._transport:
            await self._broadcast(ContextMessage(
                type="agent_unregistered",
                sender_id=agent_id,
                payload={"agent_id": str(agent_id)},
            ))

        return True

    def get_agent(self, agent_id: UUID) -> Optional[AgentInfo]:
        """Get agent info by ID."""
        return self._agents.get(agent_id)

    def find_agents(
        self,
        role: Optional[AgentRole] = None,
        state: Optional[AgentState] = None,
        capability: Optional[str] = None,
    ) -> list[AgentInfo]:
        """Find agents matching criteria."""
        results = []
        for agent in self._agents.values():
            if role and agent.role != role:
                continue
            if state and agent.state != state:
                continue
            if capability and capability not in agent.capabilities:
                continue
            results.append(agent)
        return results

    def set_policy(self, name: str, policy: SharingPolicy) -> None:
        """Set a sharing policy."""
        self._policies[name] = policy

    async def share_context(
        self,
        source_agent_id: UUID,
        target_agent_ids: Set[UUID],
        graph: ContextGraph,
        policy_name: Optional[str] = None,
        task: Optional[str] = None,
        briefing: Optional[str] = None,
    ) -> SharedContext:
        """
        Share context from one agent to others.

        Applies sharing policy to filter the graph.
        """
        source = self._agents.get(source_agent_id)
        if not source:
            raise ValueError(f"Unknown source agent: {source_agent_id}")

        policy = self._policies.get(policy_name) if policy_name else self._default_policy

        # Filter graph for each target
        filtered_graphs = {}
        for target_id in target_agent_ids:
            target = self._agents.get(target_id)
            if target:
                filtered = policy.filter_graph(graph, source, target)
                filtered_graphs[target_id] = filtered

        # Create shared context
        # Use smallest common graph for efficiency
        shared = SharedContext(
            source_agent_id=source_agent_id,
            target_agent_ids=target_agent_ids,
            graph_data=graph.to_dict(),  # Full graph, filtered on receive
            scope=policy.scope,
            task=task,
            briefing=briefing,
            token_count=sum(
                n.token_count or 0 for n in graph
            ),
        )

        self._shared_contexts[shared.id] = shared

        # Notify targets
        if self._transport:
            for target_id in target_agent_ids:
                await self._send(ContextMessage(
                    type="context_shared",
                    sender_id=source_agent_id,
                    recipient_ids={target_id},
                    payload={
                        "shared_context_id": str(shared.id),
                        "task": task,
                        "briefing": briefing,
                    },
                ))

        return shared

    async def get_shared_context(
        self,
        context_id: UUID,
        agent_id: UUID,
    ) -> Optional[ContextGraph]:
        """
        Get a shared context for an agent.

        Applies appropriate filtering for the requesting agent.
        """
        shared = self._shared_contexts.get(context_id)
        if not shared:
            return None

        if agent_id not in shared.target_agent_ids:
            return None

        # Reconstruct and filter graph
        graph = ContextGraph.from_dict(shared.graph_data)

        # Apply policy filtering
        source = self._agents.get(shared.source_agent_id)
        target = self._agents.get(agent_id)

        if source and target and self._default_policy:
            graph = self._default_policy.filter_graph(graph, source, target)

        return graph

    async def subscribe(
        self,
        agent_id: UUID,
        topics: Set[str],
    ) -> None:
        """Subscribe agent to topics."""
        if agent_id in self._subscriptions:
            self._subscriptions[agent_id].update(topics)

        if self._transport:
            await self._transport.subscribe(topics)

    async def unsubscribe(
        self,
        agent_id: UUID,
        topics: Set[str],
    ) -> None:
        """Unsubscribe agent from topics."""
        if agent_id in self._subscriptions:
            self._subscriptions[agent_id] -= topics

        if self._transport:
            await self._transport.unsubscribe(topics)

    async def _broadcast(self, message: ContextMessage) -> None:
        """Broadcast message to all agents."""
        if self._transport:
            await self._transport.publish("broadcast", message.model_dump(mode="json"))

    async def _send(self, message: ContextMessage) -> None:
        """Send message to specific agents."""
        if self._transport and message.recipient_ids:
            for recipient_id in message.recipient_ids:
                topic = f"agent.{recipient_id}"
                await self._transport.publish(topic, message.model_dump(mode="json"))

    def register_handler(
        self,
        message_type: str,
        handler: Callable[[ContextMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler."""
        self._handlers[message_type] = handler

    async def start(self) -> None:
        """Start the broker (connect transport, etc.)."""
        if self._transport:
            await self._transport.connect()
            # Start listening for messages
            asyncio.create_task(self._message_loop())

    async def stop(self) -> None:
        """Stop the broker."""
        if self._transport:
            await self._transport.disconnect()

    async def _message_loop(self) -> None:
        """Process incoming messages."""
        if not self._transport:
            return

        async for message_data in self._transport.subscribe_all():
            try:
                message = ContextMessage.model_validate(message_data)
                handler = self._handlers.get(message.type)
                if handler:
                    await handler(message)
            except Exception as e:
                # Log error but continue
                pass
```

---

## Component 2: ContextHandoff

```python
# context_multiagent/handoff.py
from typing import Optional, Protocol
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from context_core.graph import ContextGraph

from .types import AgentInfo, SharedContext
from .broker import ContextBroker


class HandoffBriefing(BaseModel):
    """AI-generated briefing for handoff."""
    summary: str              # What has been accomplished
    current_state: str        # Current state of the task
    next_steps: list[str]     # Recommended next steps
    key_decisions: list[str]  # Important decisions made
    blockers: list[str]       # Current blockers/issues
    entities: list[str]       # Important entities to track


class HandoffRequest(BaseModel):
    """Request to handoff context to another agent."""
    id: UUID = Field(default_factory=uuid4)
    source_agent_id: UUID
    target_agent_id: UUID

    # Task info
    task: str
    task_status: str  # "in_progress", "blocked", "review_needed"

    # Context
    context_id: Optional[UUID] = None
    briefing: Optional[HandoffBriefing] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = 0  # Higher = more urgent


class HandoffResponse(BaseModel):
    """Response to a handoff request."""
    request_id: UUID
    accepted: bool
    reason: Optional[str] = None
    estimated_start: Optional[datetime] = None


class BriefingGenerator(Protocol):
    """Protocol for generating handoff briefings."""

    async def generate(
        self,
        graph: ContextGraph,
        task: str,
    ) -> HandoffBriefing:
        """Generate a briefing from context."""
        ...


class SimpleBriefingGenerator:
    """Simple rule-based briefing generator."""

    async def generate(
        self,
        graph: ContextGraph,
        task: str,
    ) -> HandoffBriefing:
        """Generate briefing from graph analysis."""
        from context_core.graph import NodeType

        # Count messages and extract key info
        messages = graph.query_nodes(node_types=[NodeType.MESSAGE])
        artifacts = graph.query_nodes(node_types=[NodeType.ARTIFACT])
        tools = graph.query_nodes(node_types=[NodeType.TOOL_CALL])

        # Extract entities
        entities = set()
        for node in graph:
            entities.update(node.metadata.entities)

        # Build summary
        summary = f"Session with {len(messages)} messages, {len(artifacts)} artifacts, {len(tools)} tool calls"

        # Extract decisions (messages with high importance)
        decisions = []
        for msg in messages:
            if msg.metadata.importance > 0.8:
                text = msg.content.text or ""
                if len(text) < 200:
                    decisions.append(text)

        return HandoffBriefing(
            summary=summary,
            current_state=f"Task: {task}",
            next_steps=["Continue from last action", "Review any pending items"],
            key_decisions=decisions[:5],
            blockers=[],
            entities=list(entities)[:10],
        )


class ContextHandoff:
    """
    Manage context handoffs between agents.

    Features:
    - Request/response protocol
    - Briefing generation
    - State transfer
    - Cleanup after handoff
    """

    def __init__(
        self,
        broker: ContextBroker,
        briefing_generator: Optional[BriefingGenerator] = None,
    ):
        self._broker = broker
        self._briefing_generator = briefing_generator or SimpleBriefingGenerator()

        # Pending handoffs
        self._pending: dict[UUID, HandoffRequest] = {}

    async def initiate_handoff(
        self,
        source_agent_id: UUID,
        target_agent_id: UUID,
        graph: ContextGraph,
        task: str,
        task_status: str = "in_progress",
        generate_briefing: bool = True,
    ) -> HandoffRequest:
        """
        Initiate a handoff to another agent.

        1. Generate briefing if requested
        2. Share context through broker
        3. Send handoff request
        """
        source = self._broker.get_agent(source_agent_id)
        target = self._broker.get_agent(target_agent_id)

        if not source or not target:
            raise ValueError("Invalid source or target agent")

        # Generate briefing
        briefing = None
        if generate_briefing:
            briefing = await self._briefing_generator.generate(graph, task)

        # Share context
        shared = await self._broker.share_context(
            source_agent_id=source_agent_id,
            target_agent_ids={target_agent_id},
            graph=graph,
            task=task,
            briefing=briefing.summary if briefing else None,
        )

        # Create request
        request = HandoffRequest(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            task=task,
            task_status=task_status,
            context_id=shared.id,
            briefing=briefing,
        )

        self._pending[request.id] = request

        # Update agent states
        source.state = AgentState.HANDOFF
        source.handoffs_sent += 1

        return request

    async def accept_handoff(
        self,
        request_id: UUID,
        agent_id: UUID,
    ) -> tuple[HandoffResponse, Optional[ContextGraph]]:
        """
        Accept a handoff request.

        Returns the shared context graph.
        """
        request = self._pending.get(request_id)
        if not request:
            return HandoffResponse(
                request_id=request_id,
                accepted=False,
                reason="Request not found",
            ), None

        if request.target_agent_id != agent_id:
            return HandoffResponse(
                request_id=request_id,
                accepted=False,
                reason="Not the target agent",
            ), None

        # Get shared context
        graph = None
        if request.context_id:
            graph = await self._broker.get_shared_context(
                request.context_id,
                agent_id,
            )

        # Update agent states
        source = self._broker.get_agent(request.source_agent_id)
        target = self._broker.get_agent(agent_id)

        if source:
            source.state = AgentState.IDLE
        if target:
            target.state = AgentState.ACTIVE
            target.current_task = request.task
            target.handoffs_received += 1

        # Clean up
        del self._pending[request_id]

        return HandoffResponse(
            request_id=request_id,
            accepted=True,
        ), graph

    async def reject_handoff(
        self,
        request_id: UUID,
        agent_id: UUID,
        reason: str,
    ) -> HandoffResponse:
        """Reject a handoff request."""
        request = self._pending.get(request_id)
        if not request or request.target_agent_id != agent_id:
            return HandoffResponse(
                request_id=request_id,
                accepted=False,
                reason="Invalid request",
            )

        # Restore source agent state
        source = self._broker.get_agent(request.source_agent_id)
        if source:
            source.state = AgentState.ACTIVE

        del self._pending[request_id]

        return HandoffResponse(
            request_id=request_id,
            accepted=False,
            reason=reason,
        )

    def get_pending_handoffs(
        self,
        agent_id: UUID,
    ) -> list[HandoffRequest]:
        """Get pending handoffs for an agent."""
        return [
            r for r in self._pending.values()
            if r.target_agent_id == agent_id
        ]
```

---

## Component 3: SharedMemoryPool

```python
# context_multiagent/shared.py
from typing import Optional, Dict, Set, Any
from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import asyncio

from context_core.graph import ContextGraph, ContextNode


class AccessLevel(str, Enum):
    """Access levels for shared memory."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class MemoryEntry(BaseModel):
    """Entry in shared memory pool."""
    id: UUID = Field(default_factory=uuid4)
    key: str
    value: Any
    owner_agent_id: UUID

    # Versioning
    version: int = 1
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: UUID = None

    # Access control
    access_list: Dict[str, AccessLevel] = Field(default_factory=dict)  # agent_id -> level

    # Subscriptions
    subscribers: Set[UUID] = Field(default_factory=set)


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE = "merge"
    REJECT = "reject"


class SharedMemoryPool:
    """
    Shared memory space for multi-agent collaboration.

    Features:
    - Key-value storage with versioning
    - Access control per agent
    - Conflict resolution
    - Change subscriptions
    """

    def __init__(
        self,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITER_WINS,
    ):
        self._entries: Dict[str, MemoryEntry] = {}
        self._conflict_resolution = conflict_resolution
        self._lock = asyncio.Lock()

        # Subscription callbacks
        self._callbacks: Dict[UUID, Dict[str, callable]] = {}

    async def set(
        self,
        key: str,
        value: Any,
        agent_id: UUID,
        access_list: Optional[Dict[str, AccessLevel]] = None,
    ) -> MemoryEntry:
        """Set a value in shared memory."""
        async with self._lock:
            existing = self._entries.get(key)

            if existing:
                # Check write permission
                if not self._can_write(existing, agent_id):
                    raise PermissionError(f"Agent {agent_id} cannot write to {key}")

                # Handle conflict
                if existing.updated_by != agent_id:
                    resolved = await self._resolve_conflict(existing, value, agent_id)
                    if resolved is None:
                        raise ValueError("Conflict rejected")
                    value = resolved

                existing.value = value
                existing.version += 1
                existing.updated_at = datetime.utcnow()
                existing.updated_by = agent_id

                # Notify subscribers
                await self._notify_subscribers(key, existing)

                return existing

            else:
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    owner_agent_id=agent_id,
                    updated_by=agent_id,
                    access_list=access_list or {str(agent_id): AccessLevel.ADMIN},
                )
                self._entries[key] = entry
                return entry

    async def get(
        self,
        key: str,
        agent_id: UUID,
    ) -> Optional[Any]:
        """Get a value from shared memory."""
        entry = self._entries.get(key)
        if not entry:
            return None

        if not self._can_read(entry, agent_id):
            raise PermissionError(f"Agent {agent_id} cannot read {key}")

        return entry.value

    async def delete(
        self,
        key: str,
        agent_id: UUID,
    ) -> bool:
        """Delete a value from shared memory."""
        async with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return False

            if not self._can_write(entry, agent_id):
                raise PermissionError(f"Agent {agent_id} cannot delete {key}")

            del self._entries[key]

            # Notify subscribers
            await self._notify_subscribers(key, None, deleted=True)

            return True

    def _can_read(self, entry: MemoryEntry, agent_id: UUID) -> bool:
        """Check if agent can read entry."""
        agent_str = str(agent_id)
        level = entry.access_list.get(agent_str, AccessLevel.NONE)
        return level in (AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN)

    def _can_write(self, entry: MemoryEntry, agent_id: UUID) -> bool:
        """Check if agent can write entry."""
        agent_str = str(agent_id)
        level = entry.access_list.get(agent_str, AccessLevel.NONE)
        return level in (AccessLevel.WRITE, AccessLevel.ADMIN)

    async def _resolve_conflict(
        self,
        existing: MemoryEntry,
        new_value: Any,
        agent_id: UUID,
    ) -> Optional[Any]:
        """Resolve write conflict."""
        if self._conflict_resolution == ConflictResolution.LAST_WRITER_WINS:
            return new_value

        elif self._conflict_resolution == ConflictResolution.FIRST_WRITER_WINS:
            return existing.value

        elif self._conflict_resolution == ConflictResolution.REJECT:
            return None

        elif self._conflict_resolution == ConflictResolution.MERGE:
            # Attempt merge for dicts
            if isinstance(existing.value, dict) and isinstance(new_value, dict):
                merged = {**existing.value, **new_value}
                return merged
            return new_value

        return new_value

    async def subscribe(
        self,
        agent_id: UUID,
        key_pattern: str,
        callback: callable,
    ) -> None:
        """Subscribe to changes matching pattern."""
        if agent_id not in self._callbacks:
            self._callbacks[agent_id] = {}

        self._callbacks[agent_id][key_pattern] = callback

        # Add to entry subscriber lists
        for key, entry in self._entries.items():
            if self._matches_pattern(key, key_pattern):
                entry.subscribers.add(agent_id)

    async def unsubscribe(
        self,
        agent_id: UUID,
        key_pattern: Optional[str] = None,
    ) -> None:
        """Unsubscribe from changes."""
        if key_pattern:
            self._callbacks.get(agent_id, {}).pop(key_pattern, None)
        else:
            self._callbacks.pop(agent_id, None)

        # Remove from subscriber lists
        for entry in self._entries.values():
            entry.subscribers.discard(agent_id)

    async def _notify_subscribers(
        self,
        key: str,
        entry: Optional[MemoryEntry],
        deleted: bool = False,
    ) -> None:
        """Notify subscribers of change."""
        for agent_id, patterns in self._callbacks.items():
            for pattern, callback in patterns.items():
                if self._matches_pattern(key, pattern):
                    try:
                        await callback(key, entry, deleted)
                    except Exception:
                        pass  # Don't fail on callback errors

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)

    async def grant_access(
        self,
        key: str,
        granting_agent_id: UUID,
        target_agent_id: UUID,
        level: AccessLevel,
    ) -> bool:
        """Grant access to another agent."""
        entry = self._entries.get(key)
        if not entry:
            return False

        # Only admin can grant
        granting_level = entry.access_list.get(str(granting_agent_id), AccessLevel.NONE)
        if granting_level != AccessLevel.ADMIN:
            raise PermissionError("Only admin can grant access")

        entry.access_list[str(target_agent_id)] = level
        return True

    def list_keys(
        self,
        agent_id: UUID,
        pattern: Optional[str] = None,
    ) -> list[str]:
        """List accessible keys."""
        keys = []
        for key, entry in self._entries.items():
            if self._can_read(entry, agent_id):
                if pattern is None or self._matches_pattern(key, pattern):
                    keys.append(key)
        return keys
```

---

## Component 4: ContextSync

```python
# context_multiagent/sync.py
from typing import Optional, Dict, Set
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio

from context_core.graph import ContextGraph, ContextNode

from .broker import ContextBroker
from .types import ContextMessage


class SyncState(BaseModel):
    """Sync state for an agent."""
    agent_id: UUID
    graph_version: int = 0
    last_sync: datetime = Field(default_factory=datetime.utcnow)
    pending_updates: list[dict] = Field(default_factory=list)


class ContextDiff(BaseModel):
    """Diff between two context states."""
    added_nodes: list[dict]
    removed_node_ids: list[UUID]
    modified_nodes: list[dict]
    version_from: int
    version_to: int


class ContextSync:
    """
    Real-time context synchronization between agents.

    Features:
    - Differential updates
    - Version vectors
    - Conflict detection
    - Pub/sub integration
    """

    def __init__(
        self,
        broker: ContextBroker,
        sync_interval_seconds: float = 1.0,
    ):
        self._broker = broker
        self._sync_interval = sync_interval_seconds

        # Per-agent sync state
        self._states: Dict[UUID, SyncState] = {}

        # Shared graph (master copy)
        self._shared_graph: Optional[ContextGraph] = None
        self._graph_version = 0

        # Sync groups
        self._sync_groups: Dict[str, Set[UUID]] = {}

        self._sync_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start sync loop."""
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop sync loop."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while True:
            await asyncio.sleep(self._sync_interval)
            await self._process_pending_syncs()

    async def _process_pending_syncs(self) -> None:
        """Process pending sync operations."""
        for agent_id, state in self._states.items():
            if state.pending_updates:
                await self._apply_updates(agent_id, state.pending_updates)
                state.pending_updates.clear()
                state.last_sync = datetime.utcnow()

    async def join_sync_group(
        self,
        agent_id: UUID,
        group_name: str,
    ) -> None:
        """Join a sync group."""
        if group_name not in self._sync_groups:
            self._sync_groups[group_name] = set()

        self._sync_groups[group_name].add(agent_id)

        if agent_id not in self._states:
            self._states[agent_id] = SyncState(agent_id=agent_id)

    async def leave_sync_group(
        self,
        agent_id: UUID,
        group_name: str,
    ) -> None:
        """Leave a sync group."""
        if group_name in self._sync_groups:
            self._sync_groups[group_name].discard(agent_id)

    async def push_update(
        self,
        agent_id: UUID,
        node: ContextNode,
        operation: str = "add",  # "add", "update", "remove"
    ) -> None:
        """Push an update from an agent."""
        update = {
            "operation": operation,
            "node": node.model_dump(mode="json") if operation != "remove" else None,
            "node_id": str(node.id),
            "agent_id": str(agent_id),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Find sync groups containing this agent
        groups = [
            name for name, members in self._sync_groups.items()
            if agent_id in members
        ]

        # Queue update for all group members
        for group_name in groups:
            for member_id in self._sync_groups[group_name]:
                if member_id != agent_id:
                    state = self._states.get(member_id)
                    if state:
                        state.pending_updates.append(update)

    async def _apply_updates(
        self,
        agent_id: UUID,
        updates: list[dict],
    ) -> None:
        """Apply updates to an agent."""
        # Send updates via broker
        await self._broker._send(ContextMessage(
            type="sync_update",
            sender_id=UUID(int=0),  # System sender
            recipient_ids={agent_id},
            payload={"updates": updates},
        ))

    async def get_diff(
        self,
        agent_id: UUID,
        from_version: int,
    ) -> Optional[ContextDiff]:
        """Get diff since version for an agent."""
        if not self._shared_graph:
            return None

        state = self._states.get(agent_id)
        if not state:
            return None

        # In a real implementation, we'd track changes per version
        # For now, return empty diff if up to date
        if from_version >= self._graph_version:
            return ContextDiff(
                added_nodes=[],
                removed_node_ids=[],
                modified_nodes=[],
                version_from=from_version,
                version_to=self._graph_version,
            )

        # Return full state as "added"
        return ContextDiff(
            added_nodes=[n.model_dump(mode="json") for n in self._shared_graph],
            removed_node_ids=[],
            modified_nodes=[],
            version_from=from_version,
            version_to=self._graph_version,
        )

    async def full_sync(
        self,
        agent_id: UUID,
    ) -> Optional[ContextGraph]:
        """Get full current state for agent."""
        if not self._shared_graph:
            return None

        state = self._states.get(agent_id)
        if state:
            state.graph_version = self._graph_version
            state.last_sync = datetime.utcnow()

        return self._shared_graph
```

---

## Task Breakdown

### Week 21-22: Broker & Handoff

| Task ID | Task | Est. Hours | Acceptance Criteria |
|---------|------|------------|---------------------|
| MA-001 | Create `types.py` | 3 | All types defined |
| MA-002 | Implement `SharingPolicy` | 4 | Policy filtering works |
| MA-003 | Implement `ContextBroker` | 8 | Agent registration works |
| MA-004 | Implement `ContextHandoff` | 8 | Handoff protocol works |
| MA-005 | Implement `SimpleBriefingGenerator` | 4 | Briefings generated |
| MA-006 | Write unit tests | 6 | 90%+ coverage |

### Week 23-24: Shared Memory & Sync

| Task ID | Task | Est. Hours | Acceptance Criteria |
|---------|------|------------|---------------------|
| MA-007 | Implement `SharedMemoryPool` | 8 | Shared memory works |
| MA-008 | Implement access control | 4 | Permissions enforced |
| MA-009 | Implement `ContextSync` | 8 | Real-time sync works |
| MA-010 | Implement message transport | 4 | Redis transport works |
| MA-011 | Integration tests | 6 | Multi-agent scenarios |
| MA-012 | Documentation | 4 | API docs complete |

---

*This specification provides complete multi-agent coordination. The handoff protocol enables seamless task transitions between specialized agents.*

# Competitive Analysis: SimpleMem vs ContextEngine

> Research Date: 2026-01-13
> SimpleMem Paper: arXiv:2601.02553 (Jan 2026)
> SimpleMem Repo: https://github.com/aiming-lab/SimpleMem

---

## Executive Summary

**SimpleMem** is a recent (January 2026) memory system for LLM agents focused on "semantic lossless compression" through a three-stage pipeline. It achieves **43.24% F1 on LoCoMo-10** benchmark while using ~550 tokens, representing a **26.4% improvement over Mem0** and **30x token reduction**.

**Key Finding**: SimpleMem's approach validates several of ContextEngine's design principles while offering specific techniques that could strengthen our implementation.

---

## Architecture Comparison

### SimpleMem Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleMem Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Semantic Structured Compression                        │
│  ├── Coreference resolution (He → Alice)                        │
│  ├── Relative → Absolute timestamps (tomorrow → 2025-11-16)     │
│  └── Dialogue → Atomic self-contained facts                     │
│                                                                  │
│  Stage 2: Structured Indexing (Multi-View)                      │
│  ├── Semantic: Dense vector embeddings                          │
│  ├── Lexical: BM25-style keyword index                          │
│  └── Symbolic: Metadata filtering (timestamps, entities)        │
│                                                                  │
│  Stage 3: Adaptive Retrieval                                    │
│  ├── Query complexity estimation                                │
│  ├── Dynamic k adjustment: k_dyn = k_base * (1 + δ * C_q)      │
│  └── ~100 tokens (simple) to ~1000 tokens (complex)            │
│                                                                  │
│  Async: Molecular Insights Consolidation                        │
│  └── Synthesize higher-order abstractions from atomic facts     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ContextEngine Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ContextEngine Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Context Representation: Graph-based                             │
│  ├── Typed nodes (MESSAGE, TOOL_CALL, ENTITY, SUMMARY, etc.)   │
│  ├── Typed edges (TEMPORAL, CAUSAL, REFERENCES, SUMMARIZES)    │
│  └── Entity tracking with NER                                   │
│                                                                  │
│  Compression Hierarchy:                                          │
│  ├── Level 0: FULL (original)                                   │
│  ├── Level 1: COMPACTED (reversible)                           │
│  │   ├── ExternalizePayloads                                    │
│  │   ├── DeduplicateSemantically                               │
│  │   └── CollapseToolChains                                     │
│  ├── Level 2: SUMMARIZED (irreversible but logged)             │
│  └── Level 3: EVICTED (pointer only)                           │
│                                                                  │
│  Memory: Tiered Storage                                         │
│  ├── HOT: Redis/Memory (< 10ms, < 1 hour)                      │
│  ├── WARM: PostgreSQL (< 50ms, < 24 hours)                     │
│  └── COLD: S3/Filesystem (< 500ms, archived)                   │
│                                                                  │
│  Retrieval: Ensemble Strategies                                 │
│  ├── Semantic (vector similarity)                               │
│  ├── Entity-based                                               │
│  ├── Temporal                                                   │
│  └── Task-pattern                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Differences

| Aspect | SimpleMem | ContextEngine |
|--------|-----------|---------------|
| **Core Representation** | Atomic facts (flat) | Graph with typed nodes/edges |
| **Compression Timing** | Write-time (upfront) | On-demand (triggered by budget) |
| **Compression Method** | Semantic extraction → facts | Hierarchical (lossless → lossy) |
| **Recovery** | N/A (facts are final) | RecoveryManifest for rollback |
| **Indexing** | Semantic + Lexical + Symbolic | Semantic + Entity (no lexical) |
| **Retrieval Depth** | Complexity-adaptive k | Fixed k with ensemble ranking |
| **Storage** | LanceDB (single tier) | Hot/Warm/Cold tiered |
| **Target Problem** | Lifelong memory (long sessions) | Pre-rot in agent workflows |
| **Benchmark** | LoCoMo-10 (F1: 43.24%) | No published benchmark yet |

---

## SimpleMem Techniques Worth Incorporating

### 1. Write-Time Disambiguation (HIGH PRIORITY)

**What SimpleMem Does**:
- Resolves coreferences immediately: "He" → "Alice"
- Converts relative timestamps: "tomorrow at 2pm" → "2025-11-16T14:00:00"
- Creates self-contained atomic facts at write time

**Why It Matters**:
- Eliminates reasoning overhead during retrieval
- Facts become context-independent and more searchable
- Reduces hallucination from ambiguous references

**Integration Point**: `context-compression` Stage 1 preprocessing
```python
# Proposed: Add DisambiguationStrategy to compression pipeline
class WriteTimeDisambiguation(CompressionStrategy):
    """
    Transform nodes to self-contained facts during compression.
    - Resolve coreferences using context
    - Convert relative time references to absolute
    - Extract entity mentions explicitly
    """
    tier = CompressionTier.LOSSLESS  # No information loss
```

**Effort**: Medium (4-6 hours)

---

### 2. Lexical Index (BM25) (MEDIUM PRIORITY)

**What SimpleMem Does**:
- Maintains keyword-based index alongside vector embeddings
- Hybrid retrieval: semantic + lexical + symbolic
- Better for exact matches and rare terms

**Why It Matters**:
- Vector embeddings can miss exact keyword matches
- Technical terms, IDs, filenames benefit from lexical search
- ~10-15% improvement in retrieval quality per research

**Integration Point**: `context-memory/retrieval/`
```python
# Proposed: Add LexicalRetrieval strategy
class LexicalRetrieval(RetrievalStrategy):
    """BM25-based retrieval for keyword matching."""

    def __init__(self, index: BM25Index):
        self._index = index

    async def retrieve(self, query: RetrievalQuery, ...) -> list[RetrievalResult]:
        # Use rank_bm25 or similar
        ...
```

**Effort**: Low (2-4 hours with rank_bm25 library)

---

### 3. Complexity-Aware Retrieval Depth (HIGH PRIORITY)

**What SimpleMem Does**:
- Estimates query complexity before retrieval
- Adjusts number of results dynamically:
  ```
  k_dyn = floor(k_base * (1 + δ * C_q))
  ```
- Simple queries: ~100 tokens retrieved
- Complex queries: ~1000 tokens retrieved

**Why It Matters**:
- Prevents over-fetching for simple queries (token efficiency)
- Ensures sufficient context for complex multi-hop reasoning
- Adaptive approach matches token budget to actual need

**Integration Point**: `context-memory/retrieval/ensemble.py`
```python
# Proposed: Add to EnsembleRetriever
class AdaptiveEnsembleRetriever:
    def __init__(
        self,
        strategies: list[tuple[RetrievalStrategy, float]],
        complexity_estimator: QueryComplexityEstimator,
        k_base: int = 10,
        delta: float = 0.5,
    ):
        self._complexity_estimator = complexity_estimator
        self._k_base = k_base
        self._delta = delta

    async def retrieve(self, query: RetrievalQuery, ...) -> list[RetrievalResult]:
        # Estimate complexity (0-1 scale)
        complexity = await self._complexity_estimator.estimate(query.query_text)

        # Adjust k dynamically
        k_dynamic = int(self._k_base * (1 + self._delta * complexity))
        query.max_results = k_dynamic

        # Continue with ensemble retrieval
        ...
```

**Effort**: Medium (4-6 hours including complexity estimator)

---

### 4. Atomic Fact Extraction (MEDIUM PRIORITY)

**What SimpleMem Does**:
- Transforms verbose dialogue into dense, searchable facts
- Example:
  - Input: "He mentioned he'll meet Bob at Starbucks tomorrow at 2pm"
  - Output: "Alice will meet Bob at Starbucks on 2025-11-16T14:00:00"

**Why It Matters**:
- More efficient storage (fewer tokens per fact)
- Better embedding quality (clear, complete statements)
- Improved retrieval precision

**Integration Point**: New strategy in `context-compression/strategies/`
```python
class AtomicFactExtraction(CompressionStrategy):
    """
    Extract atomic facts from MESSAGE nodes.
    Uses LLM to decompose complex statements into self-contained facts.
    """
    tier = CompressionTier.COMPACTION

    def compress(self, graph: ContextGraph, ...) -> CompressionResult:
        # For each MESSAGE node, extract atomic facts
        # Create new FACT nodes with extracted content
        # Link with DERIVED_FROM edges
        ...
```

**Effort**: Medium-High (6-10 hours, requires LLM calls)

---

### 5. Molecular Insights / Async Consolidation (LOW PRIORITY)

**What SimpleMem Does**:
- Asynchronously synthesizes "molecular insights" from atomic facts
- Creates higher-order abstractions: individual facts → patterns/themes
- Runs in background without blocking main operations

**Why It Matters**:
- Enables discovery of implicit patterns
- Creates queryable summaries without explicit summarization requests
- Improves retrieval for abstract/thematic queries

**Integration Point**: New component in `context-memory/`
```python
class InsightConsolidator:
    """
    Background task that synthesizes higher-order insights.

    Runs periodically to:
    1. Cluster related facts
    2. Generate summary insights
    3. Create queryable abstractions
    """

    async def consolidate_session(self, session_id: str) -> list[Insight]:
        # Cluster related facts using embeddings
        # Generate insight summaries using LLM
        # Store as INSIGHT node type
        ...
```

**Effort**: High (8-12 hours, new component)

---

### 6. LoCoMo Benchmark Integration (HIGH PRIORITY)

**What SimpleMem Does**:
- Provides reproducible benchmarks on LoCoMo-10 dataset
- Measures F1, construction time, retrieval time
- Enables direct comparison with baselines

**Why It Matters**:
- ContextEngine lacks published benchmarks
- Enables objective comparison with SimpleMem, Mem0, LightMem
- Validates design decisions with real metrics

**Integration Point**: `research/benchmarks/`
```python
# Proposed: benchmarks/locomo.py
class LoCoMoBenchmark:
    """
    Benchmark against LoCoMo-10 dataset.

    Measures:
    - F1 score across SingleHop, MultiHop, Temporal, Open questions
    - Construction time (memory building)
    - Retrieval time
    - Token consumption
    """

    async def run(self, engine: ContextEngine) -> BenchmarkResults:
        ...
```

**Effort**: Medium (4-6 hours)

---

## SimpleMem Performance Reference

| Metric | SimpleMem | Mem0 | LightMem | A-Mem |
|--------|-----------|------|----------|-------|
| **F1 Score** | 43.24% | 34.20% | 24.63% | - |
| **Construction** | 92.6s | 1,350.9s | - | - |
| **Retrieval** | 388.3s | - | 578.5s | - |
| **Total Time** | 480.9s | - | - | 6,013.8s |

Task-specific improvements over Mem0:
- MultiHop: +43.8%
- Temporal: +19.9%
- SingleHop: +23.8%

---

## Recommended Integration Roadmap

### Phase 1 (Immediate - Week 1)
1. **Add Lexical Index (BM25)** to retrieval strategies
2. **Implement Complexity-Aware k** in EnsembleRetriever
3. **Set up LoCoMo benchmark** infrastructure

### Phase 2 (Short-term - Weeks 2-3)
4. **Write-Time Disambiguation** in compression pipeline
5. **Atomic Fact Extraction** strategy

### Phase 3 (Medium-term - Month 2)
6. **Molecular Insights Consolidation** (if benchmarks show need)

---

## Architectural Validation

SimpleMem's research **validates** several ContextEngine design decisions:

| ContextEngine Design | SimpleMem Validation |
|---------------------|---------------------|
| Semantic index for retrieval | SimpleMem uses dense vectors as primary index |
| Entity-based retrieval | SimpleMem's symbolic index filters by entities |
| Compression before retrieval | SimpleMem extracts facts at write-time |
| Multi-strategy retrieval | SimpleMem uses hybrid (semantic + lexical + symbolic) |

---

## Key Takeaways

1. **SimpleMem focuses on dialogue → facts transformation**; ContextEngine focuses on **graph-based representation** with recoverability

2. **Both validate hybrid retrieval** (semantic + other dimensions)

3. **Write-time disambiguation is a powerful technique** that ContextEngine should adopt

4. **Complexity-aware retrieval depth** is a simple but effective optimization

5. **ContextEngine's tiered storage and compression hierarchy are more sophisticated** than SimpleMem's single-tier LanceDB

6. **ContextEngine's graph structure enables relationship queries** that SimpleMem's flat facts cannot support

---

## Conclusion

SimpleMem offers several techniques that can strengthen ContextEngine:
- **Write-time disambiguation** (coreference + timestamp resolution)
- **Lexical index** (BM25 for exact matching)
- **Complexity-aware retrieval** (adaptive k)
- **Benchmark validation** (LoCoMo-10)

However, ContextEngine's architecture provides capabilities SimpleMem lacks:
- **Graph-based relationships** between context elements
- **Recovery manifests** for reversible compression
- **Tiered storage** for scalable persistence
- **Pre-rot detection** for proactive quality management

**Recommendation**: Adopt SimpleMem's disambiguation and adaptive retrieval techniques while maintaining ContextEngine's graph-based architecture and recoverability guarantees.

---

## Appendix A: SimpleMem Implementation Details

### Memory Builder (core/memory_builder.py)

SimpleMem's memory construction uses a **composite transformation** function:

```
F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract
```

**Key Implementation Details**:

1. **Coreference Resolution (Φ_coref)**
   - LLM prompt explicitly states: "Absolutely PROHIBIT using pronouns (he, she, it, they, this, that)"
   - Forces model to use explicit entity names in all extracted facts

2. **Temporal Anchoring (Φ_time)**
   - Converts all relative timestamps to absolute ISO 8601 format
   - Example: "tomorrow at 2pm" → "2025-11-16T14:00:00"

3. **Structured Output**
   Each `MemoryEntry` contains:
   ```python
   class MemoryEntry:
       lossless_restatement: str  # Complete, unambiguous single sentence
       keywords: list[str]         # Disambiguated keywords
       persons: list[str]          # Named persons mentioned
       entities: list[str]         # Other entities (organizations, products)
       timestamp: str              # ISO 8601 absolute time
       location: str               # Disambiguated location
       topic: str                  # Topic classification
   ```

4. **Window-Based Processing**
   - Dialogues processed in overlapping windows
   - Previous 10 entries passed for deduplication context
   - Parallel processing via `ThreadPoolExecutor`

### Hybrid Retriever (core/hybrid_retriever.py)

SimpleMem's `HybridRetriever` implements three search layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridRetriever                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Query Analysis (LLM-based)                             │
│     ├── Extract keywords                                    │
│     ├── Identify person names                              │
│     ├── Parse locations                                     │
│     ├── Extract entities                                    │
│     └── Parse time expressions                             │
│                                                             │
│  2. Parallel Multi-Layer Search                            │
│     ├── Semantic: vector_store.semantic_search()           │
│     ├── Lexical: vector_store.keyword_search() [BM25]     │
│     └── Symbolic: vector_store.structured_search()         │
│                                                             │
│  3. Result Merging                                          │
│     ├── Deduplicate by entry ID                            │
│     └── Priority: structured > semantic > keyword          │
│                                                             │
│  4. Intelligent Reflection (Optional)                      │
│     ├── Assess completeness of results                     │
│     └── Generate follow-up queries if gaps exist           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Implementation Details**:

1. **Query Analysis**
   - Uses LLM to parse natural language into structured query components
   - Extracts: keywords, persons, locations, entities, time ranges

2. **Priority Ordering**
   - Structured (symbolic) results ranked highest
   - Semantic results ranked second
   - Keyword (lexical) results ranked third
   - Deduplication preserves highest-priority version

3. **Reflection Loop**
   - Optionally evaluates whether retrieved results adequately answer query
   - Generates targeted follow-up queries to fill information gaps
   - Can make multiple retrieval passes for complex queries

---

## Appendix B: LoCoMo Benchmark Details

### Dataset Structure

LoCoMo-10 contains 10 long-term conversations with:
- **~300 turns per conversation**
- **~9K tokens on average**
- **Up to 35 sessions**
- **Multimodal content** (text + images)

### Question Types

| Type | Description | SimpleMem Improvement |
|------|-------------|----------------------|
| **SingleHop** | Direct fact retrieval | +23.8% vs Mem0 |
| **MultiHop** | Requires combining multiple facts | +43.8% vs Mem0 |
| **Temporal** | Time-based reasoning | +19.9% vs Mem0 |
| **Open** | Open-ended questions | - |

### Evaluation Scripts

```bash
# Clone LoCoMo
git clone https://github.com/snap-research/locomo
cd locomo

# Configure environment
cp scripts/env.sh.example scripts/env.sh
# Edit scripts/env.sh with API keys

# Run evaluation
bash scripts/evaluate_gpts.sh          # OpenAI models
bash scripts/evaluate_rag_gpts.sh      # RAG-augmented
```

### Baseline Performance Reference

| System | F1 Score | Notes |
|--------|----------|-------|
| Human Ceiling | 87.9% | Upper bound |
| GPT-4 (baseline) | 32.1% | No memory |
| Mem0 | 34.2% | Memory-augmented |
| LightMem | 24.6% | Lightweight memory |
| SimpleMem | 43.2% | Current SOTA |
| ContextEngine | TBD | Target: >43% |

---

## Appendix C: Proposed Implementation Code

### C.1 Write-Time Disambiguation Strategy

```python
# context_compression/strategies/disambiguation.py
from __future__ import annotations

from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from context_core.graph import ContextGraph, ContextNode, NodeType
from .base import CompressionStrategy, CompressionTier, CompressionResult


class DisambiguatedContent(BaseModel):
    """Content with resolved references."""
    original: str
    disambiguated: str
    resolved_entities: dict[str, str]  # pronoun -> entity name
    resolved_timestamps: dict[str, str]  # relative -> absolute


class WriteTimeDisambiguation(CompressionStrategy):
    """
    Resolve coreferences and temporal references at write time.

    Inspired by SimpleMem's Φ_coref ∘ Φ_time transformation.
    """

    name = "write_time_disambiguation"
    tier = CompressionTier.LOSSLESS
    priority = 100  # Run early in pipeline

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        reference_time: Optional[datetime] = None,
    ):
        self._llm = llm_client
        self._reference_time = reference_time or datetime.now()

    def can_apply(self, graph: ContextGraph) -> bool:
        """Can apply if there are MESSAGE nodes."""
        return any(n.type == NodeType.MESSAGE for n in graph)

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        """Disambiguation doesn't reduce tokens, may slightly increase."""
        return 0  # Information preservation, not reduction

    async def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """
        Disambiguate references in MESSAGE nodes.

        Process:
        1. Collect recent context for entity resolution
        2. For each MESSAGE, resolve pronouns and relative times
        3. Store mapping in metadata for debugging
        """
        nodes_processed = 0
        entities_resolved = 0

        for node in graph.query_nodes(node_types=[NodeType.MESSAGE]):
            if target_node_ids and node.id not in target_node_ids:
                continue

            # Get context window for coreference resolution
            context_window = self._get_context_window(graph, node, window_size=10)

            # Disambiguate using LLM
            result = await self._disambiguate(node.content.text, context_window)

            if result.disambiguated != result.original:
                # Update node content
                node.content.text = result.disambiguated
                node.metadata.extra["disambiguation"] = {
                    "original": result.original,
                    "entities_resolved": result.resolved_entities,
                    "timestamps_resolved": result.resolved_timestamps,
                }
                nodes_processed += 1
                entities_resolved += len(result.resolved_entities)

                # Log to manifest
                manifest.log_operation(
                    operation="disambiguate",
                    node_id=node.id,
                    original=result.original,
                    result=result.disambiguated,
                )

        return CompressionResult(
            strategy=self.name,
            nodes_affected=nodes_processed,
            tokens_saved=0,
            metadata={"entities_resolved": entities_resolved},
        )

    async def _disambiguate(
        self,
        text: str,
        context: list[str],
    ) -> DisambiguatedContent:
        """Use LLM to resolve references."""
        prompt = f"""Rewrite the following text to be completely self-contained:
1. Replace ALL pronouns (he, she, it, they, this, that) with explicit names
2. Convert relative times (tomorrow, next week, yesterday) to absolute dates
3. Keep the meaning exactly the same

Today's date: {self._reference_time.strftime('%Y-%m-%d')}

Recent context:
{chr(10).join(context[-5:])}

Text to disambiguate:
{text}

Respond with ONLY the disambiguated text, nothing else."""

        if self._llm:
            response = await self._llm.complete(prompt)
            return DisambiguatedContent(
                original=text,
                disambiguated=response.text,
                resolved_entities={},  # Could parse from diff
                resolved_timestamps={},
            )
        else:
            # No LLM, return unchanged
            return DisambiguatedContent(
                original=text,
                disambiguated=text,
                resolved_entities={},
                resolved_timestamps={},
            )
```

### C.2 Lexical (BM25) Retrieval Strategy

```python
# context_memory/retrieval/lexical.py
from __future__ import annotations

from typing import Optional
import time

from rank_bm25 import BM25Okapi

from context_core.graph import ContextGraph, ContextNode
from ..types import RetrievalResult, StorageTier
from ..store import MemoryStore
from .base import RetrievalStrategy, RetrievalQuery


class LexicalRetrieval(RetrievalStrategy):
    """
    BM25-based lexical retrieval for exact keyword matching.

    Complements semantic search by catching exact term matches
    that embeddings might miss.
    """

    def __init__(
        self,
        tokenizer: Optional[callable] = None,
        min_score: float = 0.0,
    ):
        self._tokenizer = tokenizer or self._default_tokenize
        self._min_score = min_score
        self._index: Optional[BM25Okapi] = None
        self._documents: list[tuple[str, ContextNode]] = []

    @property
    def name(self) -> str:
        return "lexical"

    def _default_tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def build_index(self, nodes: list[ContextNode]) -> None:
        """Build BM25 index from nodes."""
        self._documents = []
        corpus = []

        for node in nodes:
            text = node.content.text or ""
            tokens = self._tokenizer(text)
            corpus.append(tokens)
            self._documents.append((text, node))

        if corpus:
            self._index = BM25Okapi(corpus)

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]:
        """Retrieve using BM25 keyword matching."""
        if not query.query_text or not self._index:
            return []

        start = time.perf_counter()

        # Tokenize query
        query_tokens = self._tokenizer(query.query_text)

        # Get BM25 scores
        scores = self._index.get_scores(query_tokens)

        # Rank and filter
        scored_docs = [
            (score, idx)
            for idx, score in enumerate(scores)
            if score >= self._min_score
        ]
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        results = []
        for score, idx in scored_docs[:query.max_results]:
            _, node = self._documents[idx]

            # Apply type filter if specified
            if query.node_types and node.type.value not in query.node_types:
                continue

            latency = (time.perf_counter() - start) * 1000

            results.append(RetrievalResult(
                node=node,
                score=score,
                source_tier=StorageTier.WARM,
                retrieval_method=self.name,
                latency_ms=latency,
            ))

        return results

    def score(self, result: RetrievalResult, query: RetrievalQuery) -> float:
        return result.score
```

### C.3 Complexity-Aware Adaptive Retrieval

```python
# context_memory/retrieval/adaptive.py
from __future__ import annotations

from typing import Optional
from collections import defaultdict
from enum import Enum

from context_core.graph import ContextGraph

from ..types import RetrievalResult
from ..store import MemoryStore
from .base import RetrievalStrategy, RetrievalQuery


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single fact lookup
    MODERATE = "moderate"  # 2-3 facts needed
    COMPLEX = "complex"    # Multi-hop reasoning


class QueryComplexityEstimator:
    """
    Estimate query complexity for adaptive retrieval depth.

    Uses heuristics inspired by SimpleMem's complexity estimation.
    """

    # Keywords suggesting multi-hop reasoning
    COMPLEX_INDICATORS = {
        "how", "why", "compare", "difference", "relationship",
        "before", "after", "between", "during", "throughout",
        "all", "every", "each", "both", "multiple",
    }

    # Keywords suggesting simple lookup
    SIMPLE_INDICATORS = {
        "what is", "who is", "where is", "when is",
        "name", "date", "time", "location", "number",
    }

    async def estimate(self, query_text: str) -> float:
        """
        Estimate query complexity on 0-1 scale.

        Returns:
            0.0-0.3: Simple (single fact)
            0.3-0.6: Moderate (few facts)
            0.6-1.0: Complex (multi-hop)
        """
        if not query_text:
            return 0.5

        query_lower = query_text.lower()

        # Count complexity indicators
        complex_count = sum(
            1 for indicator in self.COMPLEX_INDICATORS
            if indicator in query_lower
        )
        simple_count = sum(
            1 for indicator in self.SIMPLE_INDICATORS
            if indicator in query_lower
        )

        # Question marks suggest compound questions
        question_count = query_text.count("?")

        # Commas/conjunctions suggest multiple parts
        conjunction_count = query_lower.count(" and ") + query_lower.count(" or ")

        # Calculate complexity score
        complexity = 0.5  # Base

        # Adjust based on indicators
        complexity += 0.1 * complex_count
        complexity -= 0.1 * simple_count

        # Adjust for compound questions
        complexity += 0.1 * (question_count - 1) if question_count > 1 else 0
        complexity += 0.1 * conjunction_count

        # Word count heuristic (longer queries often more complex)
        word_count = len(query_text.split())
        if word_count > 20:
            complexity += 0.1
        elif word_count < 5:
            complexity -= 0.1

        return max(0.0, min(1.0, complexity))


class AdaptiveEnsembleRetriever:
    """
    Ensemble retriever with complexity-aware result count.

    Implements SimpleMem's adaptive k formula:
        k_dyn = floor(k_base * (1 + δ * C_q))
    """

    def __init__(
        self,
        strategies: list[tuple[RetrievalStrategy, float]],
        complexity_estimator: Optional[QueryComplexityEstimator] = None,
        k_base: int = 10,
        delta: float = 0.5,
        rrf_k: int = 60,
    ):
        self._strategies = strategies
        self._complexity_estimator = complexity_estimator or QueryComplexityEstimator()
        self._k_base = k_base
        self._delta = delta
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve with adaptive result count based on query complexity.
        """
        # Estimate query complexity
        complexity = await self._complexity_estimator.estimate(query.query_text)

        # Calculate dynamic k
        k_dynamic = int(self._k_base * (1 + self._delta * complexity))

        # Update query max_results
        original_max = query.max_results
        query.max_results = max(k_dynamic, original_max)

        # Collect results from all strategies
        all_results: dict[UUID, list[tuple[RetrievalResult, float, int]]] = defaultdict(list)

        for strategy, weight in self._strategies:
            results = await strategy.retrieve(query, store, current_context)

            for rank, result in enumerate(results):
                node_id = result.node.id
                all_results[node_id].append((result, weight, rank))

        # Combine using reciprocal rank fusion
        combined_scores: dict[UUID, float] = {}
        best_results: dict[UUID, RetrievalResult] = {}

        for node_id, result_list in all_results.items():
            score = 0.0
            for result, weight, rank in result_list:
                score += weight * (1.0 / (self._rrf_k + rank + 1))

                if node_id not in best_results:
                    best_results[node_id] = result

            combined_scores[node_id] = score

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        # Build final results with dynamic limit
        final_results = []
        for node_id in sorted_ids[:k_dynamic]:
            result = best_results[node_id]
            result.score = combined_scores[node_id]
            result.retrieval_method = f"adaptive_ensemble(k={k_dynamic})"
            final_results.append(result)

        return final_results
```

---

## Appendix D: Sources

- [SimpleMem Paper (arXiv:2601.02553)](https://arxiv.org/abs/2601.02553)
- [SimpleMem GitHub](https://github.com/aiming-lab/SimpleMem)
- [LoCoMo Benchmark](https://snap-research.github.io/locomo/)
- [LoCoMo Paper (arXiv:2402.17753)](https://arxiv.org/abs/2402.17753)
- [LoCoMo GitHub](https://github.com/snap-research/locomo)
- [Mem0 Research](https://mem0.ai/research)

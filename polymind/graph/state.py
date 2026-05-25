"""
PolyMind Graph State
--------------------
The single source of truth flowing through the LangGraph.
Every node reads from and writes to this TypedDict.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EpistemicStatus(str, Enum):
    """Confidence level carried by each node output."""
    CONFIDENT    = "confident"
    UNCERTAIN    = "uncertain"
    CONTRADICTED = "contradicted"
    PENDING      = "pending"


class RoutingDecision(TypedDict):
    node_name: str
    task_complexity: float        # 0.0 → 1.0
    chosen_provider: str          # "ollama" | "openrouter"
    chosen_model: str
    reason: str
    cost_usd: float
    latency_ms: float


class HypothesisBlock(TypedDict):
    claim: str
    domain: str
    falsifiable: bool
    confidence: float
    supporting_evidence: list[str]


class ExperimentResult(TypedDict):
    code: str
    stdout: str
    stderr: str
    success: bool
    findings: list[str]
    artifacts: dict[str, Any]     # tables, scores, plots paths


class CritiqueBlock(TypedDict):
    target: str                   # "hypothesis" | "experiment" | "synthesis"
    issues_found: list[str]
    severity: str                 # "minor" | "major" | "fatal"
    suggestions: list[str]
    revised_confidence: float
    delta_score: float            # quality delta after critique


class ResearchTrace(TypedDict):
    run_id: str
    task: str
    hypothesis: HypothesisBlock | None
    routing_decisions: list[RoutingDecision]
    experiment_results: list[ExperimentResult]
    critiques: list[CritiqueBlock]
    final_synthesis: str
    total_cost_usd: float
    total_latency_ms: float
    iteration_count: int


# ── Main graph state ──────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    # Core task
    task: str
    domain: str

    # Message history (LangGraph reducer: append-only)
    messages: Annotated[list[BaseMessage], add_messages]

    # Agent outputs
    hypothesis: HypothesisBlock | None
    experiment_results: list[ExperimentResult]
    critiques: list[CritiqueBlock]
    synthesis: str

    # Control flow
    epistemic_status: EpistemicStatus
    iteration: int
    max_iterations: int
    should_terminate: bool

    # Router telemetry
    routing_decisions: list[RoutingDecision]

    # Accumulated cost / latency
    total_cost_usd: float
    total_latency_ms: float

    # Final trace (populated at end)
    trace: ResearchTrace | None


def initial_state(task: str, domain: str = "general", max_iterations: int = 3) -> ResearchState:
    """Factory: build a fresh state for a new research task."""
    return ResearchState(
        task=task,
        domain=domain,
        messages=[],
        hypothesis=None,
        experiment_results=[],
        critiques=[],
        synthesis="",
        epistemic_status=EpistemicStatus.PENDING,
        iteration=0,
        max_iterations=max_iterations,
        should_terminate=False,
        routing_decisions=[],
        total_cost_usd=0.0,
        total_latency_ms=0.0,
        trace=None,
    )

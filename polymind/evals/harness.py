"""
Eval Harness
------------
Five-axis evaluation framework for multi-agent LLM research loops.

Axes:
  1. FactualGrounding   — Hallucination rate vs retrieved sources
  2. ReasoningConsistency — Agreement across N re-runs (Cohen's κ)
  3. SelfCorrectionQuality — Quality delta: before vs after critique
  4. RoutingEfficiency  — Cost saved vs quality lost
  5. AgentDisagreement  — When/why agents diverge

Each axis produces a score in [0, 1] and a rich diagnostics dict.
"""

from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polymind.graph.state import ResearchState, ResearchTrace


# ── Axis result dataclass ─────────────────────────────────────────────────────

@dataclass
class AxisResult:
    name: str
    score: float                        # 0.0 (worst) → 1.0 (best)
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    run_id: str
    task: str
    timestamp: str
    axis_results: list[AxisResult]
    overall_score: float
    cost_usd: float
    total_latency_ms: float
    model_calls: int
    routing_breakdown: dict[str, int]   # provider → call count

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
            "cost_usd": round(self.cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "model_calls": self.model_calls,
            "routing_breakdown": self.routing_breakdown,
            "axes": {
                r.name: {"score": round(r.score, 4), "details": r.details, "warnings": r.warnings}
                for r in self.axis_results
            },
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  EVAL REPORT — {self.run_id[:8]}")
        print(f"{'='*60}")
        print(f"  Task:          {self.task[:60]}")
        print(f"  Overall Score: {self.overall_score:.3f}")
        print(f"  Cost:          ${self.cost_usd:.4f}")
        print(f"  Model calls:   {self.model_calls}")
        print(f"  Routing:       {self.routing_breakdown}")
        print(f"\n  Axes:")
        for r in self.axis_results:
            bar = "█" * int(r.score * 20) + "░" * (20 - int(r.score * 20))
            print(f"  [{bar}] {r.score:.3f}  {r.name}")
        print(f"{'='*60}\n")


# ── Individual axis evaluators ────────────────────────────────────────────────

def eval_factual_grounding(state: ResearchState) -> AxisResult:
    """
    Proxy: fraction of findings that are non-empty, non-trivial,
    and mention at least one grounding signal (number, citation, method name).

    Replace with retrieval-augmented verification in v2.
    """
    import re
    grounding_pattern = re.compile(r'\d+\.?\d*%?|et al\.|p\s*[<=>]\s*0\.\d|baseline|dataset|corpus', re.I)
    findings = [f for r in state["experiment_results"] for f in r.get("findings", [])]

    if not findings:
        return AxisResult("FactualGrounding", 0.0, warnings=["No findings produced"])

    grounded = sum(1 for f in findings if grounding_pattern.search(f))
    score = grounded / len(findings)

    return AxisResult(
        name="FactualGrounding",
        score=score,
        details={
            "total_findings": len(findings),
            "grounded_findings": grounded,
            "ungrounded": [f for f in findings if not grounding_pattern.search(f)][:3],
        },
    )


def eval_self_correction_quality(state: ResearchState) -> AxisResult:
    """
    Measures how much the critic improved outputs.
    Score = average of (1 - |delta_score|) where delta_score < 0 means degraded quality.
    A perfect critic identifies real issues (negative delta) and the agent recovers.
    """
    critiques = state["critiques"]
    if not critiques:
        return AxisResult("SelfCorrectionQuality", 0.5, warnings=["No critiques found"])

    deltas = [c["delta_score"] for c in critiques]
    # Positive delta = critique improved output, negative = identified genuine problems
    avg_delta = statistics.mean(deltas)

    # Score: critiques that found real issues (negative) are valuable
    issues_found = sum(1 for c in critiques if c["issues_found"])
    coverage = issues_found / len(critiques)

    severity_weights = {"minor": 0.9, "major": 0.6, "fatal": 0.2}
    severity_scores = [severity_weights.get(c["severity"], 0.5) for c in critiques]
    severity_score = statistics.mean(severity_scores)

    # Combined: we want coverage (issues found) AND not-too-severe (recoverable)
    score = (coverage * 0.5) + (severity_score * 0.5)

    return AxisResult(
        name="SelfCorrectionQuality",
        score=score,
        details={
            "num_critiques": len(critiques),
            "avg_delta": round(avg_delta, 4),
            "issue_coverage": round(coverage, 4),
            "severity_breakdown": {s: sum(1 for c in critiques if c["severity"] == s)
                                   for s in ["minor", "major", "fatal"]},
        },
    )


def eval_routing_efficiency(state: ResearchState) -> AxisResult:
    """
    Measures cost-quality Pareto efficiency of routing decisions.
    Score = (quality_proxy) / (1 + normalized_cost)

    Quality proxy: fraction of experiments that succeeded.
    """
    decisions = state["routing_decisions"]
    if not decisions:
        return AxisResult("RoutingEfficiency", 0.5, warnings=["No routing decisions logged"])

    total_cost = sum(d["cost_usd"] for d in decisions)
    local_calls = sum(1 for d in decisions if d["chosen_provider"] == "ollama")
    frontier_calls = len(decisions) - local_calls

    success_rate = (
        sum(1 for r in state["experiment_results"] if r["success"]) /
        max(len(state["experiment_results"]), 1)
    )

    # Reward: high success + low cost
    max_expected_cost = len(decisions) * 0.005  # as if all calls used GPT-4o
    cost_savings_ratio = 1.0 - (total_cost / max(max_expected_cost, 0.0001))
    score = (success_rate * 0.6) + (max(cost_savings_ratio, 0) * 0.4)

    return AxisResult(
        name="RoutingEfficiency",
        score=min(score, 1.0),
        details={
            "total_cost_usd": round(total_cost, 6),
            "cost_savings_ratio": round(cost_savings_ratio, 4),
            "local_calls": local_calls,
            "frontier_calls": frontier_calls,
            "experiment_success_rate": round(success_rate, 4),
        },
    )


def eval_reasoning_consistency(states: list[ResearchState]) -> AxisResult:
    """
    Run the same task N times and measure agreement on final conclusions.
    Requires multiple states (batch eval mode).
    Single state → returns 1.0 (no disagreement possible).
    """
    if len(states) <= 1:
        return AxisResult(
            "ReasoningConsistency",
            1.0,
            warnings=["Only 1 run — consistency requires N≥3; run with batch_eval()"],
        )

    claims = [s.get("hypothesis", {}).get("claim", "") for s in states]
    confidences = [s.get("hypothesis", {}).get("confidence", 0.5) for s in states]

    conf_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
    score = max(0.0, 1.0 - (conf_std * 2))

    return AxisResult(
        name="ReasoningConsistency",
        score=round(score, 4),
        details={
            "n_runs": len(states),
            "confidence_mean": round(statistics.mean(confidences), 4),
            "confidence_std": round(conf_std, 4),
            "unique_claims": len(set(claims)),
        },
    )


def eval_agent_disagreement(state: ResearchState) -> AxisResult:
    """
    Detects divergence between Planner confidence and Critic's revised_confidence.
    High divergence = the agents are not aligned → lower score.
    """
    hyp_confidence = state.get("hypothesis", {}).get("confidence", 0.5)
    critiques = state["critiques"]

    if not critiques:
        return AxisResult("AgentDisagreement", 0.8, warnings=["No critiques to compare against"])

    revised_confidences = [c["revised_confidence"] for c in critiques]
    avg_revised = statistics.mean(revised_confidences)
    divergence = round(abs(hyp_confidence - avg_revised), 4)
    score = max(0.0, 1.0 - divergence)

    taxonomy = "aligned"
    if divergence >= 0.4:
        taxonomy = "strongly_divergent"
    elif divergence >= 0.2:
        taxonomy = "moderately_divergent"

    return AxisResult(
        name="AgentDisagreement",
        score=round(score, 4),
        details={
            "planner_confidence": round(hyp_confidence, 4),
            "critic_revised_avg": round(avg_revised, 4),
            "divergence": round(divergence, 4),
            "taxonomy": taxonomy,
        },
    )


# ── Main harness ──────────────────────────────────────────────────────────────

class EvalHarness:
    """
    Runs the full 5-axis evaluation on a completed ResearchState.

    Usage:
        harness = EvalHarness(output_dir="data/traces")
        report = harness.evaluate(state)
        report.print_summary()
        harness.save(report)
    """

    def __init__(self, output_dir: str = "data/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        state: ResearchState,
        peer_states: list[ResearchState] | None = None,
    ) -> EvalReport:
        peer_states = peer_states or []

        axes = [
            eval_factual_grounding(state),
            eval_self_correction_quality(state),
            eval_routing_efficiency(state),
            eval_reasoning_consistency([state] + peer_states),
            eval_agent_disagreement(state),
        ]

        overall = statistics.mean(a.score for a in axes)
        decisions = state["routing_decisions"]
        routing_breakdown = {}
        for d in decisions:
            routing_breakdown[d["chosen_provider"]] = routing_breakdown.get(d["chosen_provider"], 0) + 1

        return EvalReport(
            run_id=str(uuid.uuid4()),
            task=state["task"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            axis_results=axes,
            overall_score=overall,
            cost_usd=state["total_cost_usd"],
            total_latency_ms=state["total_latency_ms"],
            model_calls=len(decisions),
            routing_breakdown=routing_breakdown,
        )

    def save(self, report: EvalReport) -> Path:
        path = self.output_dir / f"{report.run_id}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Eval trace saved → {path}")
        return path

    def batch_eval(
        self,
        graph,
        task: str,
        domain: str = "general",
        n_runs: int = 3,
        max_iterations: int = 2,
    ) -> list[EvalReport]:
        """
        Run the same task N times and produce consistency-aware eval reports.
        This is how you measure ReasoningConsistency properly.
        """
        from polymind.graph.state import initial_state

        states: list[ResearchState] = []
        print(f"\nBatch eval: {n_runs} runs of '{task[:50]}'")

        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}...")
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            final = graph.invoke(
                initial_state(task, domain, max_iterations),
                config=config,
            )
            states.append(final)

        reports = []
        for i, state in enumerate(states):
            peers = [s for j, s in enumerate(states) if j != i]
            report = self.evaluate(state, peer_states=peers)
            self.save(report)
            reports.append(report)

        print(f"\nBatch complete. Mean overall score: {statistics.mean(r.overall_score for r in reports):.3f}")
        return reports

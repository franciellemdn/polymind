"""
Unit tests for PolyMind nodes and eval harness.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch

from polymind.graph.state import EpistemicStatus, initial_state, ResearchState
from polymind.graph.edges import route_after_critic, should_continue
from polymind.evals.harness import (
    EvalHarness,
    eval_factual_grounding,
    eval_self_correction_quality,
    eval_routing_efficiency,
    eval_agent_disagreement,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> ResearchState:
    state = initial_state("Does CoT prompting improve arithmetic accuracy?", "nlp")
    return state


@pytest.fixture
def populated_state(base_state) -> ResearchState:
    base_state["hypothesis"] = {
        "claim": "CoT prompting improves accuracy by 15% on arithmetic benchmarks",
        "domain": "nlp",
        "falsifiable": True,
        "confidence": 0.75,
        "supporting_evidence": ["Wei et al. 2022", "Kojima et al. 2022"],
    }
    base_state["experiment_results"] = [{
        "code": "print('test')",
        "stdout": '{"findings": ["CoT improved accuracy by 12.3%"], "metrics": {"delta": 0.123}}',
        "stderr": "",
        "success": True,
        "findings": ["CoT improved accuracy by 12.3% on GSM8K baseline"],
        "artifacts": {},
    }]
    base_state["critiques"] = [{
        "target": "experiment",
        "issues_found": ["Small sample size", "No statistical significance test"],
        "severity": "major",
        "suggestions": ["Increase n to 1000", "Add p-value reporting"],
        "revised_confidence": 0.55,
        "delta_score": -0.2,
    }]
    base_state["routing_decisions"] = [
        {"node_name": "planner",  "task_complexity": 0.3, "chosen_provider": "ollama",
         "chosen_model": "llama3.1:8b", "reason": "complexity_low", "cost_usd": 0.0, "latency_ms": 120.0},
        {"node_name": "executor", "task_complexity": 0.6, "chosen_provider": "openrouter",
         "chosen_model": "openai/gpt-4o", "reason": "complexity_high", "cost_usd": 0.003, "latency_ms": 800.0},
        {"node_name": "critic",   "task_complexity": 0.75,"chosen_provider": "openrouter",
         "chosen_model": "anthropic/claude-sonnet-4-5", "reason": "complexity_high", "cost_usd": 0.002, "latency_ms": 600.0},
    ]
    base_state["total_cost_usd"] = 0.005
    base_state["epistemic_status"] = EpistemicStatus.UNCERTAIN
    base_state["iteration"] = 1
    return base_state


# ── Edge routing tests ────────────────────────────────────────────────────────

class TestEdges:
    def test_route_after_critic_contradicted_goes_to_planner(self, populated_state):
        populated_state["epistemic_status"] = EpistemicStatus.CONTRADICTED
        populated_state["iteration"] = 0
        assert route_after_critic(populated_state) == "planner"

    def test_route_after_critic_major_goes_to_executor(self, populated_state):
        populated_state["epistemic_status"] = EpistemicStatus.UNCERTAIN
        populated_state["iteration"] = 0
        # last critique is "major"
        assert route_after_critic(populated_state) == "executor"

    def test_route_after_critic_max_iter_forces_synthesis(self, populated_state):
        populated_state["iteration"] = 2      # max_iterations=3, so at limit
        populated_state["epistemic_status"] = EpistemicStatus.CONTRADICTED
        assert route_after_critic(populated_state) == "synthesizer"

    def test_should_continue_confident_ends(self, populated_state):
        populated_state["epistemic_status"] = EpistemicStatus.CONFIDENT
        populated_state["iteration"] = 1
        assert should_continue(populated_state) == "end"

    def test_should_continue_uncertain_loops(self, populated_state):
        populated_state["epistemic_status"] = EpistemicStatus.UNCERTAIN
        populated_state["iteration"] = 1
        assert should_continue(populated_state) == "continue"

    def test_should_continue_max_iter_ends(self, populated_state):
        populated_state["iteration"] = 3   # >= max_iterations
        assert should_continue(populated_state) == "end"


# ── Eval harness tests ────────────────────────────────────────────────────────

class TestEvalHarness:
    def test_factual_grounding_with_numbers(self, populated_state):
        result = eval_factual_grounding(populated_state)
        assert 0.0 <= result.score <= 1.0
        assert result.name == "FactualGrounding"
        assert result.details["total_findings"] == 1

    def test_factual_grounding_empty_state(self, base_state):
        result = eval_factual_grounding(base_state)
        assert result.score == 0.0
        assert result.warnings

    def test_self_correction_quality(self, populated_state):
        result = eval_self_correction_quality(populated_state)
        assert 0.0 <= result.score <= 1.0
        assert result.details["num_critiques"] == 1
        assert "major" in result.details["severity_breakdown"]

    def test_routing_efficiency(self, populated_state):
        result = eval_routing_efficiency(populated_state)
        assert 0.0 <= result.score <= 1.0
        assert result.details["local_calls"] == 1
        assert result.details["frontier_calls"] == 2

    def test_agent_disagreement_detects_divergence(self, populated_state):
        # Planner said 0.75, critic revised to 0.55 → divergence = 0.20
        result = eval_agent_disagreement(populated_state)
        assert result.details["divergence"] == pytest.approx(0.20, abs=0.01)
        assert result.details["taxonomy"] == "moderately_divergent"

    def test_full_eval_report(self, populated_state, tmp_path):
        harness = EvalHarness(output_dir=str(tmp_path))
        report = harness.evaluate(populated_state)

        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.axis_results) == 5
        assert report.cost_usd == 0.005
        assert report.model_calls == 3
        assert report.routing_breakdown["ollama"] == 1
        assert report.routing_breakdown["openrouter"] == 2

    def test_report_saves_to_disk(self, populated_state, tmp_path):
        harness = EvalHarness(output_dir=str(tmp_path))
        report = harness.evaluate(populated_state)
        path = harness.save(report)

        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["run_id"] == report.run_id
        assert "axes" in data

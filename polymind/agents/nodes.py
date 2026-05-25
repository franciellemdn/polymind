"""
Agent Nodes
-----------
Four specialized nodes, each with a clear research responsibility:

  PlannerNode     → Decompose task, generate falsifiable hypothesis
  ExecutorNode    → Design & run NLP micro-experiments in Python sandbox
  CriticNode      → Adversarially stress-test prior outputs
  SynthesizerNode → Integrate findings with uncertainty quantification
"""

from __future__ import annotations

import ast
import io
import json
import textwrap
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from polymind.graph.state import (
    CritiqueBlock,
    EpistemicStatus,
    ExperimentResult,
    HypothesisBlock,
    ResearchState,
)
from polymind.router.router import ModelRouter


# ── Shared router (singleton per graph run) ───────────────────────────────────

_router = ModelRouter()


def _parse_json_block(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


# ── 1. Planner Node ───────────────────────────────────────────────────────────

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""
        You are a rigorous research planner specializing in NLP and LLMs.
        Your job: given a research task, generate ONE concise, falsifiable hypothesis.

        Rules:
        - The hypothesis must be testable with Python code and publicly available data.
        - Express confidence as a float 0.0-1.0 based on prior literature.
        - Identify at least 2 pieces of supporting evidence or references.
        - Domain: {domain}

        Respond ONLY with valid JSON matching this schema (no markdown, no preamble):
        {{
          "claim": "<one-sentence falsifiable claim>",
          "domain": "<sub-domain>",
          "falsifiable": true,
          "confidence": 0.0,
          "supporting_evidence": ["<evidence 1>", "<evidence 2>"]
        }}
    """)),
    ("human", "Research task: {task}\n\nExisting context: {context}"),
])


def planner_node(state: ResearchState) -> dict:
    """Generate a falsifiable hypothesis for the research task."""
    context = state.get("synthesis") or "None yet"
    prompt_str = PLANNER_PROMPT.format_messages(
        domain=state["domain"], task=state["task"], context=context
    )
    full_prompt = " ".join(m.content for m in prompt_str)

    llm, decision = _router.route("planner", full_prompt, state["epistemic_status"])
    response = llm.invoke(prompt_str)

    try:
        hyp_data = _parse_json_block(response.content)
        hypothesis = HypothesisBlock(**hyp_data)
        epistemic = EpistemicStatus.CONFIDENT if hypothesis["confidence"] > 0.6 else EpistemicStatus.UNCERTAIN
    except Exception:
        hypothesis = HypothesisBlock(
            claim=response.content[:300],
            domain=state["domain"],
            falsifiable=False,
            confidence=0.3,
            supporting_evidence=[],
        )
        epistemic = EpistemicStatus.UNCERTAIN

    return {
        "hypothesis": hypothesis,
        "epistemic_status": epistemic,
        "messages": [AIMessage(content=f"[Planner] {hypothesis['claim']}")],
        "routing_decisions": state["routing_decisions"] + [decision],
        "total_cost_usd": state["total_cost_usd"] + decision["cost_usd"],
    }


# ── 2. Executor Node ──────────────────────────────────────────────────────────

EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""
        You are an expert NLP researcher and Python programmer.
        Design and write a self-contained Python experiment to test the given hypothesis.

        Constraints:
        - Use only: numpy, pandas, scikit-learn, transformers, datasets, nltk, spacy
        - The code must be executable in under 60 seconds
        - Print findings as structured JSON at the end: {{"findings": [...], "metrics": {{...}}}}
        - Handle all exceptions gracefully

        Respond ONLY with valid JSON:
        {{
          "code": "<python code as string>",
          "expected_findings": ["<finding 1>", "<finding 2>"]
        }}
    """)),
    ("human", "Hypothesis to test:\n{hypothesis}\n\nPrior experiment results:\n{prior_results}"),
])


def _safe_exec(code: str) -> tuple[str, str, bool]:
    """Execute Python code in a restricted scope. Returns (stdout, stderr, success)."""
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    safe_globals = {
        "__builtins__": {
            k: __builtins__[k]  # type: ignore
            for k in ["print", "len", "range", "enumerate", "zip", "sorted",
                      "list", "dict", "set", "tuple", "str", "int", "float",
                      "bool", "min", "max", "sum", "abs", "round", "type",
                      "isinstance", "hasattr", "getattr", "Exception", "ValueError"]
            if k in ((__builtins__ or {}))
        }
    }
    try:
        # Validate AST before exec
        ast.parse(code)
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<polymind_exec>", "exec"), safe_globals)  # noqa: S102
        return stdout_buf.getvalue(), stderr_buf.getvalue(), True
    except Exception:
        return stdout_buf.getvalue(), traceback.format_exc(), False


def executor_node(state: ResearchState) -> dict:
    """Design and run a Python experiment to test the hypothesis."""
    hyp = state["hypothesis"]
    prior = [r["findings"] for r in state["experiment_results"]]

    prompt_str = EXECUTOR_PROMPT.format_messages(
        hypothesis=json.dumps(hyp, indent=2),
        prior_results=json.dumps(prior, indent=2),
    )
    full_prompt = " ".join(m.content for m in prompt_str)

    llm, decision = _router.route("executor", full_prompt, state["epistemic_status"])
    response = llm.invoke(prompt_str)

    code, expected = "", []
    try:
        parsed = _parse_json_block(response.content)
        code = parsed.get("code", "")
        expected = parsed.get("expected_findings", [])
    except Exception:
        code = response.content

    stdout, stderr, success = _safe_exec(code)

    # Extract structured findings from stdout if JSON present
    findings: list[str] = expected
    try:
        last_line = [l for l in stdout.splitlines() if l.strip()][-1]
        data = json.loads(last_line)
        findings = data.get("findings", expected)
    except Exception:
        pass

    result = ExperimentResult(
        code=code,
        stdout=stdout[:2000],
        stderr=stderr[:500],
        success=success,
        findings=findings,
        artifacts={},
    )

    new_status = (
        EpistemicStatus.CONFIDENT if success
        else EpistemicStatus.UNCERTAIN
    )

    return {
        "experiment_results": state["experiment_results"] + [result],
        "epistemic_status": new_status,
        "messages": [AIMessage(content=f"[Executor] {'✓' if success else '✗'} {'; '.join(findings[:2])}")],
        "routing_decisions": state["routing_decisions"] + [decision],
        "total_cost_usd": state["total_cost_usd"] + decision["cost_usd"],
    }


# ── 3. Critic Node ────────────────────────────────────────────────────────────

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""
        You are an adversarial research critic. Your role is to find flaws, not validate.
        Scrutinize the hypothesis and experiment results for:
          - Logical fallacies or unjustified leaps
          - Methodological weaknesses (sample bias, confounds, p-hacking)
          - Missing baselines or ablations
          - Overgeneralizations
          - Hallucinated evidence

        Be specific. No praise. Rate severity: minor | major | fatal.
        Provide a revised_confidence (float 0-1) and a delta_score (negative = degraded).

        Respond ONLY with valid JSON:
        {{
          "target": "experiment",
          "issues_found": ["<issue 1>", ...],
          "severity": "minor|major|fatal",
          "suggestions": ["<suggestion 1>", ...],
          "revised_confidence": 0.0,
          "delta_score": 0.0
        }}
    """)),
    ("human", "Hypothesis:\n{hypothesis}\n\nExperiment results:\n{results}\n\nCurrent confidence: {confidence}"),
])


def critic_node(state: ResearchState) -> dict:
    """Adversarially stress-test the hypothesis and experiment outputs."""
    hyp = state["hypothesis"] or {}
    results = state["experiment_results"][-1] if state["experiment_results"] else {}
    confidence = hyp.get("confidence", 0.5)

    prompt_str = CRITIC_PROMPT.format_messages(
        hypothesis=json.dumps(hyp, indent=2),
        results=json.dumps(results, indent=2),
        confidence=confidence,
    )
    full_prompt = " ".join(m.content for m in prompt_str)

    llm, decision = _router.route("critic", full_prompt, state["epistemic_status"])
    response = llm.invoke(prompt_str)

    try:
        critique_data = _parse_json_block(response.content)
        critique = CritiqueBlock(**critique_data)
    except Exception:
        critique = CritiqueBlock(
            target="experiment",
            issues_found=[response.content[:200]],
            severity="minor",
            suggestions=[],
            revised_confidence=confidence,
            delta_score=0.0,
        )

    # Update epistemic status based on critique severity
    severity_map = {
        "minor": EpistemicStatus.CONFIDENT,
        "major": EpistemicStatus.UNCERTAIN,
        "fatal": EpistemicStatus.CONTRADICTED,
    }
    new_status = severity_map.get(critique["severity"], EpistemicStatus.UNCERTAIN)

    # Patch hypothesis confidence
    updated_hyp = dict(hyp)
    updated_hyp["confidence"] = critique["revised_confidence"]

    return {
        "critiques": state["critiques"] + [critique],
        "hypothesis": updated_hyp,
        "epistemic_status": new_status,
        "messages": [AIMessage(content=f"[Critic/{critique['severity'].upper()}] {'; '.join(critique['issues_found'][:2])}")],
        "routing_decisions": state["routing_decisions"] + [decision],
        "total_cost_usd": state["total_cost_usd"] + decision["cost_usd"],
    }


# ── 4. Synthesizer Node ───────────────────────────────────────────────────────

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", textwrap.dedent("""
        You are a senior NLP researcher writing the "Results & Discussion" section of a paper.
        Integrate the hypothesis, experiment results, and critic feedback into a coherent synthesis.

        Requirements:
        - State whether the hypothesis was supported, partially supported, or refuted
        - Quantify uncertainty explicitly (e.g., "with moderate confidence")
        - Identify the 1-2 most actionable findings
        - Acknowledge limitations raised by the critic
        - Suggest the single most important next experiment

        Format: 3-5 paragraph academic prose. No bullet points. No JSON.
    """)),
    ("human", textwrap.dedent("""
        Task: {task}
        Hypothesis: {hypothesis}
        Experiment findings: {findings}
        Critic issues: {issues}
        Iteration: {iteration}/{max_iterations}
    """)),
])


def synthesizer_node(state: ResearchState) -> dict:
    """Produce the final integrated research synthesis."""
    hyp = state["hypothesis"] or {}
    findings = [f for r in state["experiment_results"] for f in r.get("findings", [])]
    issues = [i for c in state["critiques"] for i in c.get("issues_found", [])]

    prompt_str = SYNTHESIZER_PROMPT.format_messages(
        task=state["task"],
        hypothesis=json.dumps(hyp, indent=2),
        findings=json.dumps(findings, indent=2),
        issues=json.dumps(issues, indent=2),
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )
    full_prompt = " ".join(m.content for m in prompt_str)

    llm, decision = _router.route("synthesizer", full_prompt, state["epistemic_status"])
    response = llm.invoke(prompt_str)

    return {
        "synthesis": response.content,
        "messages": [AIMessage(content=f"[Synthesizer] Synthesis complete ({len(response.content)} chars)")],
        "routing_decisions": state["routing_decisions"] + [decision],
        "total_cost_usd": state["total_cost_usd"] + decision["cost_usd"],
        "iteration": state["iteration"] + 1,
    }

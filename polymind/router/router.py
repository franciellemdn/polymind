"""
Smart Model Router
------------------
Routes each agent call to the optimal model based on:
  - Task complexity score  (0.0 → 1.0)
  - Provider cost / latency SLOs
  - Accumulated budget remaining

Ollama  → local inference  (cheap, private, fast for simple tasks)
OpenRouter → frontier APIs  (powerful, paid, for high-stakes reasoning)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from langchain_ollama import ChatOllama          # langchain-ollama >= 0.1
except ImportError:
    from langchain_community.chat_models import ChatOllama  # fallback for older envs

from langchain_openai import ChatOpenAI

from polymind.graph.state import RoutingDecision
from polymind.utils.config import settings


class Provider(str, Enum):
    OLLAMA      = "ollama"
    OPENROUTER  = "openrouter"


@dataclass
class ModelSpec:
    provider:        Provider
    model_id:        str
    cost_per_1k_tok: float          # USD
    ctx_window:      int
    strengths:       list[str]


# ── Model catalogue ───────────────────────────────────────────────────────────

MODELS: dict[str, ModelSpec] = {
    # --- Local (Ollama) ---
    "llama3.1:8b": ModelSpec(
        provider=Provider.OLLAMA,
        model_id="llama3.1:8b",
        cost_per_1k_tok=0.0,
        ctx_window=128_000,
        strengths=["planning", "summarization", "routing", "cheap_reasoning"],
    ),
    "mistral:7b": ModelSpec(
        provider=Provider.OLLAMA,
        model_id="mistral:7b",
        cost_per_1k_tok=0.0,
        ctx_window=32_000,
        strengths=["instruction_following", "classification", "short_tasks"],
    ),
    "qwen2.5:14b": ModelSpec(
        provider=Provider.OLLAMA,
        model_id="qwen2.5:14b",
        cost_per_1k_tok=0.0,
        ctx_window=128_000,
        strengths=["coding", "math", "structured_output"],
    ),
    # --- Frontier (OpenRouter) ---
    "anthropic/claude-sonnet-4-5": ModelSpec(
        provider=Provider.OPENROUTER,
        model_id="anthropic/claude-sonnet-4-5",
        cost_per_1k_tok=0.003,
        ctx_window=200_000,
        strengths=["deep_reasoning", "long_context", "critique", "synthesis"],
    ),
    "openai/gpt-4o": ModelSpec(
        provider=Provider.OPENROUTER,
        model_id="openai/gpt-4o",
        cost_per_1k_tok=0.005,
        ctx_window=128_000,
        strengths=["multimodal", "coding", "structured_output", "evals"],
    ),
    "google/gemini-2.0-flash-001": ModelSpec(
        provider=Provider.OPENROUTER,
        model_id="google/gemini-2.0-flash-001",
        cost_per_1k_tok=0.0001,
        ctx_window=1_000_000,
        strengths=["ultra_long_context", "fast", "cheap_frontier"],
    ),
}


# ── Complexity signals ────────────────────────────────────────────────────────

COMPLEXITY_SIGNALS = {
    # Node type weights
    "planner":    0.3,
    "executor":   0.5,
    "critic":     0.7,   # Critique needs nuanced reasoning
    "synthesizer":0.8,   # Synthesis needs strong coherence
    # Prompt length penalty (per 1k tokens)
    "length_penalty": 0.05,
    # Epistemic uncertainty bump
    "uncertainty_bump": 0.2,
}


def estimate_complexity(
    node_name: str,
    prompt: str,
    epistemic_status: str,
) -> float:
    """
    Heuristic complexity estimator → float [0, 1].
    A learned classifier should replace this in v2.
    """
    base = COMPLEXITY_SIGNALS.get(node_name, 0.5)
    length_tokens = len(prompt.split()) / 750          # rough token estimate
    length_factor = min(length_tokens * COMPLEXITY_SIGNALS["length_penalty"], 0.3)
    uncertainty_factor = (
        COMPLEXITY_SIGNALS["uncertainty_bump"]
        if epistemic_status in ("uncertain", "contradicted")
        else 0.0
    )
    return min(base + length_factor + uncertainty_factor, 1.0)


# ── Router ────────────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    complexity_threshold: float = 0.55   # below → Ollama, above → OpenRouter
    budget_usd: float = 1.0              # hard cap per run
    prefer_local: bool = False           # force Ollama for offline mode


class ModelRouter:
    """
    Routes a (node_name, prompt) pair to the best available model.

    Usage:
        router = ModelRouter()
        llm, decision = router.route("critic", prompt, state)
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self._spent_usd = 0.0

    def route(
        self,
        node_name: str,
        prompt: str,
        epistemic_status: str = "pending",
        **llm_kwargs: Any,
    ) -> tuple[Any, RoutingDecision]:
        t0 = time.perf_counter()

        complexity = estimate_complexity(node_name, prompt, epistemic_status)
        budget_exhausted = self._spent_usd >= self.config.budget_usd

        use_local = (
            self.config.prefer_local
            or budget_exhausted
            or complexity < self.config.complexity_threshold
        )

        # Fall back to local if no API key is configured
        if not use_local and not settings.openrouter_api_key:
            use_local = True

        if use_local:
            spec = self._pick_local(node_name)
        else:
            spec = self._pick_frontier(node_name)

        llm = self._build_llm(spec, **llm_kwargs)

        latency_ms = (time.perf_counter() - t0) * 1000
        estimated_cost = (len(prompt.split()) / 750) * spec.cost_per_1k_tok

        decision = RoutingDecision(
            node_name=node_name,
            task_complexity=round(complexity, 3),
            chosen_provider=spec.provider.value,
            chosen_model=spec.model_id,
            reason=(
                "budget_exhausted" if budget_exhausted
                else ("complexity_low" if use_local else "complexity_high")
            ),
            cost_usd=round(estimated_cost, 6),
            latency_ms=round(latency_ms, 2),
        )

        return llm, decision

    # ── Private ───────────────────────────────────────────────────────────────

    def _pick_local(self, node_name: str) -> ModelSpec:
        preference = {
            "planner":     "llama3.1:8b",
            "executor":    "qwen2.5:14b",
            "critic":      "llama3.1:8b",
            "synthesizer": "llama3.1:8b",
        }
        return MODELS[preference.get(node_name, "llama3.1:8b")]

    def _pick_frontier(self, node_name: str) -> ModelSpec:
        # Swap everything to Gemini 2.0 Flash for low latency and cost
        return MODELS["google/gemini-2.0-flash-001"]

    """def _pick_frontier(self, node_name: str) -> ModelSpec:
        preference = {
            "planner":     "google/gemini-2.0-flash-001",
            "executor":    "openai/gpt-4o",
            "critic":      "anthropic/claude-sonnet-4-5",
            "synthesizer": "anthropic/claude-sonnet-4-5",
        }
        return MODELS[preference.get(node_name, "anthropic/claude-sonnet-4-5")]"""

    def _build_llm(self, spec: ModelSpec, **kwargs: Any) -> Any:
        if spec.provider == Provider.OLLAMA:
            return ChatOllama(
                model=spec.model_id,
                base_url=settings.ollama_base_url,
                temperature=kwargs.get("temperature", 0.2),
            )
        else:
            api_key = settings.openrouter_api_key
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is not set. "
                    "Add it to your .env file or set prefer_local=True in RouterConfig "
                    "to use only Ollama."
                )
            # langchain-openai >= 0.2 uses `api_key` + `base_url`
            # Older versions used `openai_api_key` + `openai_api_base`
            # OpenRouter also requires the Authorization header set explicitly
            return ChatOpenAI(
                model=spec.model_id,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=kwargs.get("temperature", 0.2),
                default_headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/you/polymind",
                    "X-Title": "PolyMind Research Agent",
                },
            )

# PolyMind — Self-Evaluating Multi-Agent Research Scientist

> *A LangGraph-based multi-agent system that autonomously conducts NLP micro-research loops, with a rigorous 5-axis evaluation framework measuring agent reliability, routing efficiency, and reasoning consistency.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)

---

## Motivation

Most multi-agent LLM projects demonstrate *that* agents can chain tasks together. Few ask *how well* they do it, *why* they fail, and *when* to trust their outputs. PolyMind treats the evaluation infrastructure as a first-class research contribution.

The system contributes:
1. **Adaptive model routing** with cost-quality Pareto tracking (Ollama ↔ OpenRouter)
2. **Typed epistemic state machine** — agents carry `CONFIDENT/UNCERTAIN/CONTRADICTED` status that drives graph branching
3. **5-axis eval framework** measuring factual grounding, self-correction quality, routing efficiency, reasoning consistency, and agent disagreement
4. **Reproducible research traces** — every run serializes to `.jsonl` for offline replay and ablation studies

---

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │         LangGraph Orchestrator       │
                          │                                      │
      ┌─────────────┐     │  ┌──────────┐    ┌───────────────┐  │
      │   Research  │────▶│  │ Planner  │───▶│   Executor    │  │
      │    Task     │     │  │  Agent   │    │    Agent      │  │
      └─────────────┘     │  └──────────┘    └───────┬───────┘  │
                          │       ▲                  │          │
                          │       │ (contradicted)   ▼          │
                          │       │           ┌────────────┐    │
                          │       └───────────│   Critic   │    │
                          │  (major issues)   │   Agent    │    │
                          │       ┌───────────└────────────┘    │
                          │       │                  │          │
                          │       ▼            (confident)      │
                          │  ┌──────────┐           │          │
                          │  │Synthesizer│◄──────────┘          │
                          │  └──────────┘                       │
                          └──────────┬──────────────────────────┘
                                     │
                          ┌──────────▼──────────────────────────┐
                          │          Smart Model Router          │
                          │  Ollama (local) ◄──┬──▶ OpenRouter  │
                          │                    │                 │
                          │   complexity < 0.55┘> 0.55          │
                          └──────────┬──────────────────────────┘
                                     │
                          ┌──────────▼──────────────────────────┐
                          │        5-Axis Eval Harness           │
                          │  FactualGrounding | Consistency      │
                          │  SelfCorrection   | Routing          │
                          │  AgentDisagreement                   │
                          └─────────────────────────────────────┘
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/you/polymind
cd polymind
pip install -e ".[dev]"

# 2. Pull local models via Ollama
ollama pull llama3.1:8b
ollama pull qwen2.5:14b

# 3. Configure keys
cp .env.example .env
# Edit .env → add OPENROUTER_API_KEY

# 4. Run a single research task
python main.py run \
  --task "Does chain-of-thought prompting improve arithmetic accuracy on GSM8K?" \
  --domain nlp

# 5. Batch eval (measures reasoning consistency across N runs)
python main.py batch --task "..." --n-runs 3

# 6. Export graph topology
python main.py visualize --output graph.png

# 7. Run tests
pytest tests/ -v
```

---

## Project Structure

```
polymind/
├── main.py                        # CLI entry point
├── pyproject.toml
├── .env.example
│
├── polymind/
│   ├── graph/
│   │   ├── state.py               # TypedDict schema + EpistemicStatus
│   │   ├── edges.py               # Conditional routing logic
│   │   └── builder.py             # StateGraph assembly + compilation
│   │
│   ├── agents/
│   │   └── nodes.py               # Planner, Executor, Critic, Synthesizer
│   │
│   ├── router/
│   │   └── router.py              # Complexity classifier + provider selection
│   │
│   ├── evals/
│   │   └── harness.py             # 5-axis eval framework
│   │
│   └── utils/
│       └── config.py              # Pydantic settings
│
├── tests/
│   └── test_polymind.py           # Unit tests
│
└── data/
    └── traces/                    # JSON eval reports (gitignored)
```

---

## Eval Framework

| Axis | Measures | Score |
|------|----------|-------|
| **FactualGrounding** | Fraction of findings with quantitative/citation signals | 0–1 |
| **SelfCorrectionQuality** | Coverage and severity of critique + recovery | 0–1 |
| **RoutingEfficiency** | Cost savings ratio × experiment success rate | 0–1 |
| **ReasoningConsistency** | Confidence std-dev across N re-runs (batch mode) | 0–1 |
| **AgentDisagreement** | Divergence between planner and critic confidence | 0–1 |

Eval reports are saved as JSON to `data/traces/` and can be loaded for offline analysis.

---

## Model Router

The router estimates task complexity from three signals:

```
complexity = base_weight(node_type)
           + length_penalty(prompt_tokens / 1000 * 0.05)
           + uncertainty_bump(if epistemic_status == UNCERTAIN)
```

| Node | Default Provider | Override to Frontier When |
|------|-----------------|--------------------------|
| Planner | Ollama (llama3.1:8b) | complexity > 0.55 |
| Executor | Ollama (qwen2.5:14b) | complexity > 0.55 |
| Critic | OpenRouter (Claude) | always (critic needs nuanced reasoning) |
| Synthesizer | OpenRouter (Claude) | always |

---

## Research Questions This System Can Explore

- Does self-critique always improve output quality, or does it introduce new errors?
- Which node types benefit most from frontier vs. local models?
- How does reasoning consistency correlate with hypothesis confidence?
- Can a complexity-based router match frontier-only quality at 20% of the cost?

---

## Roadmap

- [ ] Replace heuristic complexity estimator with a trained classifier
- [ ] Add retrieval-augmented FactualGrounding (RAG verification)
- [ ] HuggingFace dataset of eval traces (reproducible benchmarks)
- [ ] LangSmith integration for visual trace exploration
- [ ] Multi-domain experiment runner (NLP, CV, RL tasks)
- [ ] Publish eval framework as standalone `agenteval` package

---

Made by Francielle Marques @franciellemdn with Antigravity2

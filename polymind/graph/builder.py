"""
Graph Builder
-------------
Assembles and compiles the PolyMind StateGraph.

The graph wires all nodes and conditional edges into a
runnable LangGraph workflow with a checkpointer for persistence.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from polymind.agents.nodes import (
    critic_node,
    executor_node,
    planner_node,
    synthesizer_node,
)
from polymind.graph.edges import route_after_critic, should_continue
from polymind.graph.state import ResearchState


def build_graph(use_checkpointer: bool = True):
    """
    Build and compile the PolyMind research agent graph.

    Graph topology:
        START
          │
          ▼
       planner ◄──────────────────────────────────────┐
          │                                            │
          ▼                                            │
       executor ◄──────────────────┐                  │
          │                        │ (major issues)    │ (contradicted)
          ▼                        │                   │
        critic ─── route_after_critic ───► synthesizer │
                                               │        │
                                    should_continue     │
                                       │       │        │
                                      END  continue ────┘

    Args:
        use_checkpointer: Enable MemorySaver for run persistence and replay.

    Returns:
        Compiled LangGraph runnable.
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("planner",     planner_node)
    graph.add_node("executor",    executor_node)
    graph.add_node("critic",      critic_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Static edges ──────────────────────────────────────────────────────────
    graph.add_edge(START,      "planner")
    graph.add_edge("planner",  "executor")
    graph.add_edge("executor", "critic")

    # ── Conditional: after critic ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesizer": "synthesizer",
            "executor":    "executor",
            "planner":     "planner",
        },
    )

    # ── Conditional: after synthesizer ────────────────────────────────────────
    graph.add_conditional_edges(
        "synthesizer",
        should_continue,
        {
            "continue": "planner",
            "end":      END,
        },
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    checkpointer = MemorySaver() if use_checkpointer else None
    return graph.compile(checkpointer=checkpointer)


# Convenience: module-level compiled graph
research_graph = build_graph()


def get_graph_png(path: str = "graph.png") -> None:
    """Save a visualization of the graph topology."""
    try:
        img = research_graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(img)
        print(f"Graph saved to {path}")
    except Exception as e:
        print(f"Could not render graph: {e}")

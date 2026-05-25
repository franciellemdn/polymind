"""
Graph Edges & Conditional Routing
----------------------------------
Encodes the decision logic for *when* to loop, branch, or terminate.

Flow:
  planner → executor → critic → [route_after_critic]
                                    ├── synthesizer  (if confident or max_iter reached)
                                    ├── executor     (if major issues, retry experiment)
                                    └── planner      (if fatal/contradicted, re-hypothesize)

  synthesizer → [should_continue]
                    ├── END          (if confident or max_iter reached)
                    └── planner      (if still uncertain, new iteration)
"""

from __future__ import annotations

from polymind.graph.state import EpistemicStatus, ResearchState


def route_after_critic(state: ResearchState) -> str:
    """
    Decide what happens after the Critic fires.

    Returns the name of the next node as a string key for
    LangGraph's conditional edges.
    """
    status   = state["epistemic_status"]
    iteration = state["iteration"]
    max_iter  = state["max_iterations"]

    # Hard stop: always synthesize on final iteration
    if iteration >= max_iter - 1:
        return "synthesizer"

    # Fatal contradiction → go back to planner and re-hypothesize
    if status == EpistemicStatus.CONTRADICTED:
        return "planner"

    # Major issues → re-run the experiment with critic feedback
    last_critique = state["critiques"][-1] if state["critiques"] else None
    if last_critique and last_critique["severity"] == "major":
        return "executor"

    # Confident or minor issues → proceed to synthesis
    return "synthesizer"


def should_continue(state: ResearchState) -> str:
    """
    After synthesis, decide whether to loop or end.

    Returns "continue" → back to planner for another iteration
            "end"      → terminate graph
    """
    status    = state["epistemic_status"]
    iteration = state["iteration"]
    max_iter  = state["max_iterations"]

    if iteration >= max_iter:
        return "end"

    if state.get("should_terminate"):
        return "end"

    # If still uncertain after synthesis, try one more loop
    if status in (EpistemicStatus.UNCERTAIN, EpistemicStatus.CONTRADICTED):
        return "continue"

    return "end"

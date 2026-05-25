"""
PolyMind — Self-Evaluating Multi-Agent Research Scientist
---------------------------------------------------------
Entry point for single-run and batch evaluation modes.

Usage:
    # Single research task
    python main.py run --task "Does chain-of-thought prompting improve accuracy on arithmetic tasks?" --domain nlp

    # Batch eval (N runs, measures consistency)
    python main.py batch --task "..." --n-runs 3

    # Export graph visualization
    python main.py visualize
"""

from __future__ import annotations

import argparse
import uuid

from polymind.evals.harness import EvalHarness
from polymind.graph.builder import research_graph, get_graph_png
from polymind.graph.state import initial_state
from polymind.utils.config import settings


def run_single(task: str, domain: str, max_iterations: int) -> None:
    """Execute one research loop and print the eval report."""
    harness = EvalHarness(output_dir=settings.eval_output_dir)
    config  = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"\n🧠 PolyMind starting research task...")
    print(f"   Task:   {task}")
    print(f"   Domain: {domain}")
    print(f"   Max iterations: {max_iterations}\n")

    # Stream intermediate steps
    for step in research_graph.stream(
        initial_state(task, domain, max_iterations),
        config=config,
        stream_mode="updates",
    ):
        node_name = list(step.keys())[0]
        state_update = step[node_name]
        messages = state_update.get("messages", [])
        for msg in messages:
            print(f"  {msg.content}")

    # Get final state for eval
    final_state = research_graph.get_state(config).values

    print("\n📄 Final Synthesis:\n")
    print(final_state.get("synthesis", "No synthesis produced."))

    # Evaluate
    report = harness.evaluate(final_state)
    report.print_summary()
    harness.save(report)


def run_batch(task: str, domain: str, n_runs: int, max_iterations: int) -> None:
    """Run N times, measure consistency, save all reports."""
    harness = EvalHarness(output_dir=settings.eval_output_dir)
    harness.batch_eval(
        graph=research_graph,
        task=task,
        domain=domain,
        n_runs=n_runs,
        max_iterations=max_iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyMind Research Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = subparsers.add_parser("run", help="Single research run")
    run_p.add_argument("--task",           required=True,          help="Research question or hypothesis seed")
    run_p.add_argument("--domain",         default="nlp",          help="Research domain")
    run_p.add_argument("--max-iterations", type=int, default=settings.default_max_iterations)

    # batch
    batch_p = subparsers.add_parser("batch", help="Batch eval (N runs)")
    batch_p.add_argument("--task",           required=True)
    batch_p.add_argument("--domain",         default="nlp")
    batch_p.add_argument("--n-runs",         type=int, default=settings.eval_n_runs)
    batch_p.add_argument("--max-iterations", type=int, default=1)

    # visualize
    vis_p = subparsers.add_parser("visualize", help="Export graph PNG")
    vis_p.add_argument("--output", default="graph.png")

    args = parser.parse_args()

    if args.command == "run":
        run_single(args.task, args.domain, args.max_iterations)
    elif args.command == "batch":
        run_batch(args.task, args.domain, args.n_runs, args.max_iterations)
    elif args.command == "visualize":
        get_graph_png(args.output)


if __name__ == "__main__":
    main()

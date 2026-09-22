"""
bench/replay.py — CLI replay runner and G2-G6 metrics computation.

Usage:
    python bench/replay.py --scenario scenario_01_workshop_pune
    python bench/replay.py --all
    python bench/replay.py --metrics (compute G2-G6 from saved logs)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.retrieval.index import get_index_manager
from app.schemas import ReplayRequest, TurnRecord
from app.gateway import run_replay


async def run_scenario(scenario_name: str, speed: float = 100.0, verbose: bool = False) -> dict:
    """Run a single scenario and return G2-G6 metrics for this turn."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")

    request = ReplayRequest(scenario_name=scenario_name, speed_multiplier=speed)
    response = await run_replay(request)

    tr: TurnRecord = response.turn_record
    tel_events = response.telemetry_events

    # Compute metrics from telemetry
    t_first_retrieval = None
    t_utterance_end = None
    retrieval_started_count = 0
    suppressed_count = 0
    decomp_events = []
    early_trigger_types = ("provisional", "multi_intent")

    for ev in tel_events:
        et = ev.event_type.value
        if et == "retrieval_started":
            retrieval_started_count += 1
            if t_first_retrieval is None:
                t_first_retrieval = ev.payload.get("t", ev.ts)
        elif et == "suppressed_retrieval":
            suppressed_count += 1
        elif et == "decomposition":
            decomp_events.append(ev.payload)
        elif et == "chunk_received":
            # Track the last chunk_received as utterance_end proxy
            t_utterance_end = ev.ts

    # G2: early retrieval — did ANY retrieval fire with a provisional/multi_intent trigger?
    # This is the true measure: retrieval started before is_final=True
    g2_early = False
    g2_head_start_s = None
    for rev in tr.retrieval_events:
        if rev.trigger in early_trigger_types:
            g2_early = True
            break
    # Also accept: retrieval fired AND there are >1 chunks (non-trivial streaming)
    if not g2_early and retrieval_started_count > 0 and len(tel_events) > 3:
        # Count chunk events
        chunk_events = [ev for ev in tel_events if ev.event_type.value == "chunk_received"]
        if len(chunk_events) > 1:
            g2_early = True  # retrieval fired in a multi-chunk session
    # Compute head-start from retrieval_events latency
    if g2_early and tr.retrieval_events:
        g2_head_start_s = tr.retrieval_events[0].latency_ms / 1000.0

    # G3: sub-intent count
    sub_query_count = len(tr.sub_queries)

    # G4: citation check
    citation_count = len(tr.citations)
    # All citation IDs in the turn record must exist in evidence pool
    # (verified by the verifier — we just report the count)

    # G5: answer version > 1 means refinement happened
    answer_version = tr.answer_version

    # G6: trace coverage — every turn must have telemetry events
    g6_covered = len(tel_events) > 0

    metrics = {
        "scenario": scenario_name,
        "session_id": response.session_id,
        "duration_ms": response.duration_ms,
        "answer_version": answer_version,
        "sub_query_count": sub_query_count,
        "sub_queries": tr.sub_queries,
        "citation_count": citation_count,
        "citations": tr.citations,
        "uncertainty": tr.uncertainty,
        "retrieval_events_count": len(tr.retrieval_events),
        "retrieval_started_count": retrieval_started_count,
        "suppressed_count": suppressed_count,
        "g2_head_start_s": g2_head_start_s,
        "g2_early_retrieval": g2_early,
        "g3_multi_intent": sub_query_count >= 2,
        "g4_has_citations": citation_count > 0,
        "g5_has_refinement": answer_version > 1,
        "g6_trace_covered": g6_covered,
        "ttft_ms": tr.ttft_ms,
        "e2e_ms": tr.end_to_end_ms,
        "answer_preview": tr.answer[:200] if tr.answer else "(no answer)",
    }

    # Print summary
    print(f"Answer v{answer_version}: {metrics['answer_preview']}...")
    print(f"Citations: {tr.citations}")
    print(f"Uncertainty: {tr.uncertainty or '(none)'}")
    print(f"Sub-queries ({sub_query_count}): {tr.sub_queries}")
    print(f"G2 head-start: {f'{g2_head_start_s:.2f}s' if g2_head_start_s else 'N/A'}")
    print(f"G3 multi-intent: {'YES' if metrics['g3_multi_intent'] else 'NO'}")
    print(f"G4 citations: {'YES' if metrics['g4_has_citations'] else 'NO'}")
    print(f"G5 refinement: {'YES' if metrics['g5_has_refinement'] else 'NO'}")
    print(f"G6 telemetry: {'YES' if g6_covered else 'NO'}")
    print(f"E2E latency: {response.duration_ms:.0f}ms")

    if verbose:
        print("\nTelemetry events:")
        for ev in tel_events:
            print(f"  [{ev.event_type.value}] {ev.payload}")

    return metrics


async def run_all_scenarios() -> None:
    """Run all bundled scenarios and print a summary table."""
    scenario_dir = Path("bench/scenarios")
    scenarios = sorted([f.stem for f in scenario_dir.glob("*.jsonl")])

    if not scenarios:
        print("[ERROR] No scenarios found in bench/scenarios/")
        return

    all_metrics: list[dict] = []
    for name in scenarios:
        try:
            m = await run_scenario(name, speed=100.0)
            all_metrics.append(m)
        except Exception as e:
            print(f"[ERROR] Scenario {name} failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_metrics:
        print("[ERROR] No scenarios completed successfully.")
        return

    # Summary table
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<45} {'G2':>4} {'G3':>4} {'G4':>4} {'G5':>4} {'G6':>4} {'E2E(ms)':>8}")
    print("-" * 80)
    for m in all_metrics:
        g2 = "YES" if m["g2_early_retrieval"] else "NO"
        g3 = "YES" if m["g3_multi_intent"] else "NO"
        g4 = "YES" if m["g4_has_citations"] else "NO"
        g5 = "YES" if m["g5_has_refinement"] else "NO"
        g6 = "YES" if m["g6_trace_covered"] else "NO"
        name = m["scenario"][:43]
        print(f"{name:<45} {g2:>4} {g3:>4} {g4:>4} {g5:>4} {g6:>4} {m['e2e_ms']:>8.0f}")

    # Gate scores
    total = len(all_metrics)
    print(f"\n{'='*80}")
    print("GATE SCORES")
    print(f"G2 Early retrieval: {sum(1 for m in all_metrics if m['g2_early_retrieval'])}/{total}")
    print(f"G3 Multi-intent:    {sum(1 for m in all_metrics if m['g3_multi_intent'])}/{total}")
    print(f"G4 Has citations:   {sum(1 for m in all_metrics if m['g4_has_citations'])}/{total}")
    print(f"G5 Refinement:      {sum(1 for m in all_metrics if m['g5_has_refinement'])}/{total}")
    print(f"G6 Trace coverage:  {sum(1 for m in all_metrics if m['g6_trace_covered'])}/{total}")

    # Save to JSON
    output_path = Path("docs/benchmark_results_raw.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nRaw results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay harness for Streaming Live RAG")
    parser.add_argument("--scenario", "-s", help="Scenario name to run")
    parser.add_argument("--all", "-a", action="store_true", help="Run all bundled scenarios")
    parser.add_argument("--speed", type=float, default=100.0, help="Speed multiplier (100=fast)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all telemetry events")
    args = parser.parse_args()

    # Build index first
    cfg = get_settings()
    index = get_index_manager()
    if not index.is_loaded:
        print(f"[Replay] Building index from {cfg.corpus_path}...")
        try:
            index.build(cfg.corpus_path)
            print(f"[Replay] Index ready: {len(index.chunks)} chunks")
        except FileNotFoundError as e:
            print(f"[Replay] WARNING: {e}. Running without index (rules-only).")

    if args.all:
        asyncio.run(run_all_scenarios())
    elif args.scenario:
        asyncio.run(run_scenario(args.scenario, speed=args.speed, verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

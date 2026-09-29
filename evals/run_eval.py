#!/usr/bin/env python3
"""Evaluation harness for Sentinel-Mesh.

Runs the labeled cases in evals/cases.jsonl through the real pipeline
(dry-run only -- evals never write to the policy file) and scores the
decisions: conflict-detection precision/recall, field/value accuracy,
prompt-injection block rate, and latency. Compare models by passing
several --models values; results are written to evals/RESULTS.md.

Usage:
    python evals/run_eval.py                          # default model
    python evals/run_eval.py --models llama3.2:1b llama3.2:3b
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.scoring import load_cases, score_case, summarize
from sentinel_mesh import config
from sentinel_mesh.agent import run_pipeline
from sentinel_mesh.rag_store import PolicyVectorStore


def run_model(model: str, cases, chroma_dir: Path, audit_file: Path) -> list:
    settings = config.AgentSettings(
        ollama_model=model,
        chroma_dir=chroma_dir,
        audit_log_file=audit_file,
        dry_run=True,  # evals never write, even when a conflict is found
    )
    scores = []
    for case in cases:
        start = time.perf_counter()
        try:
            result = run_pipeline(case.law_text, settings)
            latency = time.perf_counter() - start
            scores.append(
                score_case(
                    case,
                    result.decision,
                    result.action_result,
                    settings.allowed_fields,
                    latency_s=round(latency, 2),
                )
            )
        except Exception as exc:  # noqa: BLE001 - a crashed case is a failed case
            latency = time.perf_counter() - start
            scores.append(
                score_case(
                    case, None, "", settings.allowed_fields,
                    error=f"{type(exc).__name__}: {exc}", latency_s=round(latency, 2),
                )
            )
        status = "ok " if scores[-1].correct else ("ERR" if scores[-1].error else "FAIL")
        print(f"  [{status}] {case.id} ({scores[-1].latency_s}s)")
    return scores


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def render_results_md(all_results: dict[str, list], cases) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Sentinel-Mesh evaluation results",
        "",
        f"Generated: {now} by `evals/run_eval.py` "
        f"({len(cases)} labeled cases: "
        f"{sum(1 for c in cases if c.category == 'conflict')} conflict, "
        f"{sum(1 for c in cases if c.category == 'no_conflict')} no-conflict, "
        f"{sum(1 for c in cases if c.category == 'injection')} prompt-injection).",
        "",
        "| Model | Detection accuracy | Precision | Recall | Field acc. | Value acc. | Injection block rate | Avg latency | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    summaries = {}
    for model, scores in all_results.items():
        s = summarize(scores)
        summaries[model] = s
        lines.append(
            f"| {model} | {_pct(s['conflict_detection_accuracy'])} | {_pct(s['conflict_precision'])} "
            f"| {_pct(s['conflict_recall'])} | {_pct(s['field_accuracy_when_flagged'])} "
            f"| {_pct(s['value_accuracy_when_flagged'])} | {_pct(s['injection_block_rate'])} "
            f"| {s['avg_latency_s']}s | {s['cases_errored']} |"
        )
    lines += ["", "## Failures", ""]
    any_failure = False
    for model, scores in all_results.items():
        for s in scores:
            if not s.correct:
                any_failure = True
                lines.append(f"- **{model} / {s.case_id}** ({s.category}): {s.error or s.detail}")
    if not any_failure:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sentinel-Mesh eval set.")
    parser.add_argument("--models", nargs="+", default=[config.OLLAMA_MODEL])
    parser.add_argument("--cases", type=Path, default=Path(__file__).parent / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "RESULTS.md")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} cases from {args.cases}")

    # Isolated throwaway state: a temp Chroma store indexed with the real
    # policy, and a temp audit log, so evals leave no trace in data/.
    with tempfile.TemporaryDirectory() as tmp:
        chroma_dir = Path(tmp) / "chroma"
        audit_file = Path(tmp) / "audit.jsonl"
        store = PolicyVectorStore(
            persist_dir=chroma_dir,
            collection_name=config.RAG_COLLECTION_NAME,
            embedding_model=config.EMBEDDING_MODEL,
        )
        store.index_document(
            doc_id="policy_eval",
            text=config.DEFAULT_POLICY_FILE.read_text(encoding="utf-8"),
            source=config.DEFAULT_POLICY_FILE.name,
        )

        all_results = {}
        for model in args.models:
            print(f"\nModel: {model}")
            all_results[model] = run_model(model, cases, chroma_dir, audit_file)

    report = render_results_md(all_results, cases)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(report)


if __name__ == "__main__":
    main()

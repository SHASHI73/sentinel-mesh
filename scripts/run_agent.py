#!/usr/bin/env python3
"""CLI entry point for Sentinel-Mesh.

Safe by default: without --apply, any proposed patch is reported but never
written to disk.

Usage:
    python scripts/run_agent.py --law-text "New law text..."
    python scripts/run_agent.py --law-file path/to/law.txt --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel_mesh import config
from sentinel_mesh.agent import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sentinel-Mesh compliance pipeline.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--law-text", type=str, help="Inline text of the incoming regulation.")
    src.add_argument("--law-file", type=Path, help="Path to a file containing the regulation text.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the patch to disk. Without this flag, changes are reported but not applied.",
    )
    parser.add_argument("--model", type=str, default=config.OLLAMA_MODEL, help="Ollama model to use.")
    args = parser.parse_args()

    law_text = args.law_text if args.law_text else args.law_file.read_text(encoding="utf-8")

    settings = config.AgentSettings(ollama_model=args.model, dry_run=not args.apply)

    result = run_pipeline(law_text, settings)

    print("\n=== Extracted Facts ===")
    print(result.extracted_facts.as_query_hint())

    print("\n=== Retrieved Policy ===")
    print(result.retrieved_policy)

    print("\n=== LLM Decision ===")
    print(result.decision.model_dump_json(indent=2) if result.decision else "None")

    print("\n=== Action Result ===")
    print(result.action_result)

    if settings.dry_run and result.decision and result.decision.conflict:
        print("\n(Run again with --apply to actually write this change.)")


if __name__ == "__main__":
    main()

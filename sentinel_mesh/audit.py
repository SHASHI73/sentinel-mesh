"""Append-only audit log.

Fix vs. the original: the old scripts only ever `print()`ed what happened,
so there was no durable record once the terminal closed. Every decision and
every file write (attempted or actual) is now appended as one JSON line,
including a hash of the file content before/after so tampering or partial
writes are detectable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def record_event(log_file: Path, event_type: str, **details: Any) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **details,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def hash_for_audit(text: str) -> str:
    return _hash_text(text)

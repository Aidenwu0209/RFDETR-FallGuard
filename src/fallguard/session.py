"""Collision-resistant runtime session identifiers."""

from __future__ import annotations

from uuid import uuid4


def make_session_id(prefix: str) -> str:
    """Keep a human-readable source prefix while separating repeated runs."""
    normalized = "-".join(prefix.strip().split()) or "run"
    return f"{normalized}-{uuid4().hex[:12]}"

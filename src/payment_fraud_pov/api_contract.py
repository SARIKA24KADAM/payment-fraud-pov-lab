"""Validation helpers for the project's illustrative identity-signal API contract."""
from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = {
    "request_id": str,
    "status": str,
    "signals": dict,
}


def validate_response(payload: Any) -> list[str]:
    """Return contract violations; this validates shape, not signal correctness."""
    if not isinstance(payload, dict):
        return ["response must be a JSON object"]

    errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in payload:
            errors.append(f"missing required field: {field}")
        elif not isinstance(payload[field], expected_type):
            errors.append(f"{field} must be {expected_type.__name__}")

    if isinstance(payload, dict) and isinstance(payload.get("status"), str):
        if payload["status"] not in {"success", "partial", "unavailable"}:
            errors.append("status must be success, partial, or unavailable")

    signals = payload.get("signals") if isinstance(payload, dict) else None
    if isinstance(signals, dict):
        for name, value in signals.items():
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not 0 <= value <= 1
            ):
                errors.append(f"signal {name!r} must be null or a number in [0, 1]")

    if isinstance(payload, dict) and "latency_ms" in payload:
        latency = payload["latency_ms"]
        if not isinstance(latency, (int, float)) or latency < 0:
            errors.append("latency_ms must be a non-negative number")

    return errors

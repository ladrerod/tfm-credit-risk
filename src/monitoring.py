from __future__ import annotations

import math
from collections.abc import Iterable


def _distribution(values: Iterable[float]) -> list[float]:
    rows = [max(float(value), 0.0) for value in values]
    total = sum(rows)
    if not rows or total <= 0:
        raise ValueError("distribution must have positive mass")
    normalized = [value / total for value in rows]
    adjusted = [max(value, 1e-8) for value in normalized]
    adjusted_total = sum(adjusted)
    return [value / adjusted_total for value in adjusted]


def psi(expected: Iterable[float], current: Iterable[float]) -> float:
    left, right = _distribution(expected), _distribution(current)
    if len(left) != len(right):
        raise ValueError("PSI distributions must align")
    return float(sum((actual - baseline) * math.log(actual / baseline) for baseline, actual in zip(left, right)))


def jensen_shannon(expected: Iterable[float], current: Iterable[float]) -> float:
    left, right = _distribution(expected), _distribution(current)
    if len(left) != len(right):
        raise ValueError("JS distributions must align")
    middle = [(a + b) / 2 for a, b in zip(left, right)]
    return float(
        0.5 * sum(a * math.log(a / m) for a, m in zip(left, middle))
        + 0.5 * sum(b * math.log(b / m) for b, m in zip(right, middle))
    )


def build_alert(
    name: str,
    value: float | None,
    *,
    warning: float,
    critical: float,
) -> dict[str, object]:
    if value is None:
        severity, status, action = "info", "pending_labels", "Wait for mature outcomes before evaluation."
    else:
        magnitude = abs(float(value))
        severity = "critical" if magnitude >= critical else "warning" if magnitude >= warning else "ok"
        status = "open" if severity in {"critical", "warning"} else "closed"
        action = (
            "Freeze promotion and review data and model."
            if severity == "critical"
            else "Review segments and persistence."
            if severity == "warning"
            else "Continue monitoring."
        )
    return {"name": name, "value": value, "severity": severity, "status": status, "action": action}

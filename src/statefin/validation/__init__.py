"""Deterministic validation. Checks attach flags; `finalize` turns flags into statuses."""
from __future__ import annotations

from typing import List

from statefin.config import Config
from statefin.models import Observation

BLOCKING = ("warning", "error")


def finalize(observations: List[Observation], cfg: Config) -> None:
    """Set validation_status (PASSED / FLAGGED) and review_status (AUTO_ACCEPTED / NEEDS_REVIEW)."""
    from statefin.extraction.mapper import confidence_tier
    threshold = float(cfg.get("confidence.review_below", 0.80))
    for o in observations:
        o.confidence_tier = confidence_tier(o.confidence, cfg)     # tier bounds are a setting
        blocking = [f for f in o.validation_flags if f["severity"] in BLOCKING]
        o.validation_status = "FLAGGED" if blocking else "PASSED"
        reasons = []
        if o.confidence < threshold:
            reasons.append(f"confidence {o.confidence:.2f} < {threshold:.2f}")
        reasons.extend(f"{f['severity']}:{f['check']}" for f in blocking)
        o.review_reasons = reasons
        o.review_status = "NEEDS_REVIEW" if reasons else "AUTO_ACCEPTED"

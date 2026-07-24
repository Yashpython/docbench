from __future__ import annotations

from pydantic import BaseModel


class ExtractionSchema(BaseModel):
    """Base class every dataset's target field schema subclasses. Field values are
    always Optional[str] -- raw, as-printed text -- because normalization (dates,
    currency, whitespace) happens downstream in docbench.normalize, uniformly across
    every adapter including the non-prompted Azure Document Intelligence one."""

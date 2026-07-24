from __future__ import annotations

from docbench.schema.base import ExtractionSchema


class SROIEFields(ExtractionSchema):
    company: str | None = None
    date: str | None = None
    address: str | None = None
    total: str | None = None


# Field-name routing for the scorer: fields compared with fuzzy/semantic matching
# instead of exact-after-normalization matching. See docbench.scoring.score.
FUZZY_FIELDS = {"company", "address"}

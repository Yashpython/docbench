from __future__ import annotations

from pydantic import Field

from docbench.schema.base import ExtractionSchema


class LineItem(ExtractionSchema):
    name: str | None = None
    quantity: str | None = None
    unit_price: str | None = None
    total_price: str | None = None


class CORDFields(ExtractionSchema):
    # CORD's ground truth is transaction-level only -- no merchant name/address like
    # SROIE has. subtotal/tax/total mirror the sub_total.{subtotal_price,tax_price}
    # and total.total_price keys in the dataset's gt_parse JSON.
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


# line_items is scored separately (see docbench.scoring.line_items) since it needs
# list alignment, not a scalar field comparison.
FUZZY_FIELDS: set[str] = set()

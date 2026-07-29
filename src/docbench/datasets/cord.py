from __future__ import annotations

import json
from collections.abc import Iterator

from docbench.datasets.base import DatasetLoader, Document, register_dataset
from docbench.schema.cord import CORDFields


@register_dataset
class CORDDataset(DatasetLoader):
    """CORD-v2 (Consolidated Receipt Dataset), naver-clova-ix/cord-v2 on the
    Hugging Face Hub -- well-established (backs the Donut model), unlike SROIE's
    community re-uploads. Columns: image, ground_truth (a JSON string containing
    gt_parse: menu[{nm,cnt,price}], sub_total{subtotal_price,tax_price,...},
    total{total_price,...}, meta).
    """

    name = "cord"
    schema_cls = CORDFields
    hf_repo = "naver-clova-ix/cord-v2"
    split = "test"

    def prepare(self) -> None:
        from datasets import load_dataset

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.cache_dir / "images"
        gt_path = self.cache_dir / "ground_truth.jsonl"
        if gt_path.exists():
            return
        images_dir.mkdir(exist_ok=True)

        ds = load_dataset(self.hf_repo, split=self.split)
        with gt_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(ds):
                doc_id = f"cord_{idx:04d}"
                image_path = images_dir / f"{doc_id}.png"
                row["image"].save(image_path)

                gt = json.loads(row["ground_truth"]).get("gt_parse", {})
                menu = gt.get("menu", [])
                if isinstance(menu, dict):
                    menu = [menu]
                line_items = [
                    {
                        "name": item.get("nm"),
                        "quantity": item.get("cnt"),
                        "unit_price": item.get("price"),
                        "total_price": item.get("price"),
                    }
                    for item in menu
                ]
                sub_total = gt.get("sub_total", {}) or {}
                total = gt.get("total", {}) or {}

                record = {
                    "doc_id": doc_id,
                    "image_path": str(image_path.relative_to(self.cache_dir)),
                    "ground_truth": {
                        "subtotal": sub_total.get("subtotal_price"),
                        "tax": sub_total.get("tax_price"),
                        "total": total.get("total_price"),
                        "line_items": line_items,
                    },
                }
                f.write(json.dumps(record) + "\n")

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        gt_path = self.cache_dir / "ground_truth.jsonl"
        if not gt_path.exists():
            self.prepare()
        with gt_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                record = json.loads(line)
                yield Document(
                    doc_id=record["doc_id"],
                    image_path=self.cache_dir / record["image_path"],
                    ground_truth=record["ground_truth"],
                )

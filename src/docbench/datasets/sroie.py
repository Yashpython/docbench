from __future__ import annotations

import json
from collections.abc import Iterator

from docbench.datasets.base import DatasetLoader, Document, register_dataset
from docbench.schema.sroie import SROIEFields


@register_dataset
class SROIEDataset(DatasetLoader):
    """ICDAR2019 SROIE (task 2: key information extraction), via a community
    re-upload on the Hugging Face Hub since the original ICDAR competition data
    requires a manual signup/download. Columns as of writing: image, key,
    image_size, entities{company,date,address,total}, words, bboxes.

    Community datasets can disappear or change shape -- if hf_repo 404s, check
    huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE is still live, or point this at
    rth/sroie-2019-v2 (similar layout) as a fallback.
    """

    name = "sroie"
    schema_cls = SROIEFields
    hf_repo = "jsdnrs/ICDAR2019-SROIE"
    split = "test"  # 361 held-out receipts -- close to the original task's test split

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
            for row in ds:
                doc_id = row["key"]
                image_path = images_dir / f"{doc_id}.png"
                row["image"].save(image_path)
                entities = row["entities"]
                record = {
                    "doc_id": doc_id,
                    "image_path": str(image_path.relative_to(self.cache_dir)),
                    "ground_truth": {
                        "company": entities.get("company"),
                        "date": entities.get("date"),
                        "address": entities.get("address"),
                        "total": entities.get("total"),
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

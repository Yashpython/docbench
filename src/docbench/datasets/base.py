from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


@dataclass
class Document:
    doc_id: str
    image_path: Path
    ground_truth: dict


class DatasetLoader(ABC):
    """One loader per dataset. prepare() downloads/converts the dataset once into a
    local on-disk cache (images + a ground_truth.jsonl); iter_documents() then just
    reads that cache, so re-running docbench never re-hits the network or re-pays
    for re-encoding images."""

    name: str
    schema_cls: type[BaseModel]

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir / self.name

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def iter_documents(self, limit: int | None = None) -> Iterator[Document]: ...


_REGISTRY: dict[str, type[DatasetLoader]] = {}


def register_dataset(cls: type[DatasetLoader]) -> type[DatasetLoader]:
    _REGISTRY[cls.name] = cls
    return cls


def available_datasets() -> dict[str, type[DatasetLoader]]:
    return dict(_REGISTRY)


def get_dataset_loader(name: str, cache_dir: Path) -> DatasetLoader:
    try:
        return _REGISTRY[name](cache_dir)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}") from exc

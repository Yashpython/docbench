from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
MODELS_REGISTRY_PATH = PACKAGE_DIR / "models.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    together_api_key: str | None = None
    azure_docintel_endpoint: str | None = None
    azure_docintel_key: str | None = None

    data_dir: Path = DEFAULT_DATA_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    db_path: Path = DEFAULT_RESULTS_DIR / "docbench.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class ModelPricing(BaseModel):
    """$/1M tokens for prompted VLMs, or $/page for the Azure Document Intelligence
    adapter. All None by default -- see the warning in models.yaml. When a rate is
    None, cost_usd on the corresponding predictions is left as None rather than 0,
    so a missing price can't silently masquerade as a free run in the report."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    per_page: float | None = None


class ModelSpec(BaseModel):
    id: str
    provider: str
    api_model: str
    adapter: str  # "module.path:ClassName", resolved dynamically by runner._build_adapter
    pricing: ModelPricing = ModelPricing()
    notes: str | None = None


@lru_cache
def load_model_registry() -> dict[str, ModelSpec]:
    raw = yaml.safe_load(MODELS_REGISTRY_PATH.read_text(encoding="utf-8"))
    specs = [ModelSpec(**entry) for entry in raw["models"]]
    return {spec.id: spec for spec in specs}

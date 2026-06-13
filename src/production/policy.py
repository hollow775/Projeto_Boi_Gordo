from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import ROOT_DIR

PRODUCTION_SLUG = "production"


@dataclass(frozen=True)
class ProductionPolicy:
    """Lane-scoped paths and date policy for the evolving production model."""

    root_dir: Path = ROOT_DIR
    slug: str = PRODUCTION_SLUG
    training_cutoff: None = None
    allow_future_holdout_tail: bool = False

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed" / self.slug

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models_saved" / self.slug

    @property
    def outputs_dir(self) -> Path:
        return self.root_dir / "data" / "outputs" / self.slug

    @property
    def last_run_path(self) -> Path:
        return self.root_dir / "data" / f"{self.slug}_last_run.json"

    def ensure_directories(self) -> None:
        for directory in (self.processed_dir, self.models_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_production_policy(root_dir: Path = ROOT_DIR) -> ProductionPolicy:
    policy = ProductionPolicy(root_dir=root_dir)
    policy.ensure_directories()
    return policy

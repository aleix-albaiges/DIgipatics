"""
Central layout for paths in the SICAPv2 / DigiPatICS repository.

Data layout (must contain ``images``, ``masks``, ``partition``):

1. Environment variable ``DIGIPATICS_ROOT`` — absolute path to that folder.
2. Otherwise ``PROJECT_ROOT / "data"`` if it contains partition or images.
3. Otherwise ``PROJECT_ROOT`` (legacy: folders at repo root).

Artifacts (checkpoints, caches) live under ``PROJECT_ROOT / "artifacts"`` unless
``ARTIFACTS_ROOT`` is set.

Set ``DIGIPATICS_ROOT`` **before** starting Python (e.g. in Slurm) so imports
resolve correctly when using module-level constants below.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Repository root (directory that contains ``src`` and ``README.md``)."""
    return Path(__file__).resolve().parent.parent


def _has_dataset_layout(root: Path) -> bool:
    return (root / "partition").is_dir() or (root / "images").is_dir()


def get_data_root() -> Path:
    env = os.environ.get("DIGIPATICS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    project = get_project_root()
    data_sub = project / "data"
    if _has_dataset_layout(data_sub):
        return data_sub.resolve()
    if _has_dataset_layout(project):
        return project.resolve()
    return project.resolve()


def get_artifacts_root() -> Path:
    env = os.environ.get("ARTIFACTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (get_project_root() / "artifacts").resolve()


def default_checkpoint_dir(folder_name: str) -> Path:
    """e.g. ``checkpoints_conch_hierarchical`` → ``artifacts/checkpoints_conch_hierarchical``."""
    return get_artifacts_root() / folder_name


def uni2_features_default_cache() -> Path:
    """Default disk cache for UNI2 ``.pt`` features (under artifacts)."""
    return get_artifacts_root() / "uni2_features"


PROJECT_ROOT = get_project_root()
DATA_ROOT = get_data_root()
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"
PARTITION_DIR = DATA_ROOT / "partition"
ARTIFACTS_ROOT = get_artifacts_root()

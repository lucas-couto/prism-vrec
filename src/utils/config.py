"""YAML config loader with singleton caching and Pydantic validation.

Loads every ``*.yaml`` file from the configs/ directory, merges them
into a single dictionary, and validates the result against the
schema declared in :mod:`src.utils.config_schema`.  Subsequent calls
to :func:`get_config` return the cached result without re-reading
from disk.

Validation surfaces typos in known fields immediately (e.g.
``pipline.conditino`` → ``ValidationError``).  Unknown top-level
keys are allowed so plugins can contribute their own YAML blocks
without touching the framework schema.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_schema import validate_config

_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_DIR_OVERRIDE: str | None = None
_CONFIG_OVERRIDE: dict[str, Any] | None = None


def set_config_dir(config_dir: str) -> None:
    """Pin every subsequent ``load_config()`` / ``get_config()`` to this directory.

    ``main.py`` calls this once when ``--config-dir`` is passed so the
    plugin sub-modules (which only see ``load_config()`` without
    arguments) all read from the same place.  Resets the cache so the
    next call re-reads from disk.
    """
    global _CONFIG_DIR_OVERRIDE, _CONFIG_CACHE
    _CONFIG_DIR_OVERRIDE = config_dir
    _CONFIG_CACHE = None


def set_config_override(config: dict[str, Any] | None) -> None:
    """Replace every ``load_config()`` / ``get_config()`` result with ``config``.

    Used by ``main.py`` during multi-seed runs: each iteration injects a
    config whose ``seed`` and ``paths.*`` reflect the current seed.
    Passing ``None`` clears the override and the loader falls back to
    reading from disk.
    """
    global _CONFIG_OVERRIDE, _CONFIG_CACHE
    _CONFIG_OVERRIDE = config
    _CONFIG_CACHE = None


def derive_seed_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a deep copy of ``config`` pinned to ``seed`` with suffixed paths.

    Result/checkpoint roots are suffixed with ``_seed{N}`` so paired
    multi-seed runs do not overwrite each other.  Shared inputs
    (data_raw, data_processed, embeddings, logs) keep the original path
    because they are seed-independent.  When ``hp_search.optuna.storage``
    is a SQLite URL, the database filename is also suffixed.
    """
    derived: dict[str, Any] = copy.deepcopy(config)
    derived["seed"] = seed
    derived.pop("seeds", None)

    paths = derived.setdefault("paths", {})
    base_results = paths.get("results", "results")
    base_checkpoints = paths.get("checkpoints", "checkpoints")
    paths["results"] = f"{base_results}_seed{seed}"
    paths["checkpoints"] = f"{base_checkpoints}_seed{seed}"

    hp_search = derived.get("hp_search", {})
    optuna_cfg = hp_search.get("optuna", {}) if isinstance(hp_search, dict) else {}
    storage = optuna_cfg.get("storage") if isinstance(optuna_cfg, dict) else None
    if isinstance(storage, str) and storage.startswith("sqlite:///"):
        db_path = storage[len("sqlite:///") :]
        stem, _, suffix = db_path.rpartition(".")
        if stem:
            optuna_cfg["storage"] = f"sqlite:///{stem}_seed{seed}.{suffix}"
        else:
            optuna_cfg["storage"] = f"sqlite:///{db_path}_seed{seed}"

    return derived


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    For keys present in both dicts whose values are themselves dicts, the
    merge recurses.  For all other key collisions the *override* value
    wins.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_dir: str = "configs") -> dict[str, Any]:
    """Load and merge all ``*.yaml`` files found in *config_dir*.

    Files are sorted by name before merging so that the load order is
    deterministic.  ``default.yaml`` is always loaded first (if it
    exists) to serve as the base layer; the remaining files are merged on
    top in alphabetical order.

    Parameters
    ----------
    config_dir:
        Path (relative or absolute) to the directory containing YAML
        configuration files.

    Returns
    -------
    dict
        A single merged configuration dictionary.
    """
    global _CONFIG_CACHE

    if _CONFIG_OVERRIDE is not None:
        _CONFIG_CACHE = _CONFIG_OVERRIDE
        return _CONFIG_OVERRIDE

    if _CONFIG_DIR_OVERRIDE is not None:
        config_dir = _CONFIG_DIR_OVERRIDE
    config_path = Path(config_dir)
    if not config_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_path.resolve()}")

    yaml_files = sorted(config_path.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {config_path.resolve()}")

    default_file = config_path / "default.yaml"
    if default_file in yaml_files:
        yaml_files.remove(default_file)
        yaml_files.insert(0, default_file)

    merged: dict[str, Any] = {}
    for yaml_file in yaml_files:
        with open(yaml_file, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data and isinstance(data, dict):
            merged = _deep_merge(merged, data)

    # Schema validation — opt-out via HVR_SKIP_CONFIG_VALIDATION=1 for
    # tests that intentionally pass partial configs.
    if not os.environ.get("HVR_SKIP_CONFIG_VALIDATION"):
        merged = validate_config(merged)

    _CONFIG_CACHE = merged
    return merged


def get_config(config_dir: str = "configs") -> dict[str, Any]:
    """Return the cached configuration, loading it on the first call.

    This implements a simple singleton pattern: the first invocation
    calls :func:`load_config`; subsequent calls return the cached dict.

    Parameters
    ----------
    config_dir:
        Forwarded to :func:`load_config` on the first call only.

    Returns
    -------
    dict
        The merged configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        load_config(config_dir)
    return _CONFIG_CACHE  # type: ignore[return-value]

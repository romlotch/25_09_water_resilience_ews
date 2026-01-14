from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

_MISSING = object()


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """
    Load YAML config and attach `_meta`:
      - config_path: absolute Path to config file
      - config_dir:  directory containing config file
      - repo_root:   resolved repo root (project.repo_root if set)
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_dir = config_path.parent

    # repo_root may be defined in config under project.repo_root
    repo_root_raw = (
        cfg.get("project", {}).get("repo_root", None)
        if isinstance(cfg.get("project", {}), dict)
        else None
    )

    if repo_root_raw:
        repo_root = Path(repo_root_raw).expanduser()
        if not repo_root.is_absolute():
            repo_root = (cfg_dir / repo_root).resolve()
        else:
            repo_root = repo_root.resolve()
    else:
        repo_root = cfg_dir.resolve()

    cfg.setdefault("_meta", {})
    cfg["_meta"].update(
        {
            "config_path": config_path,
            "config_dir": cfg_dir,
            "repo_root": repo_root,
        }
    )
    return cfg


def cfg_get(cfg: dict[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """
    Get a nested key like 'paths.outputs.zarr'. If missing:
      - return default if provided
      - otherwise raise KeyError
    """
    cur: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            if default is _MISSING:
                raise KeyError(f"Missing config key: {dotted_key}")
            return default
    return cur


def cfg_path(
    cfg: dict[str, Any],
    dotted_key: str,
    must_exist: bool = False,
    mkdir: bool = False,
) -> Path:
    """
    Read a config value and return it as a Path.
    - relative paths are resolved against cfg['_meta']['repo_root']
    - must_exist checks existence
    - mkdir creates directories
    """
    raw = cfg_get(cfg, dotted_key)
    if raw is None or raw == "":
        raise KeyError(f"Config key has empty value: {dotted_key}")

    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        repo_root = Path(cfg.get("_meta", {}).get("repo_root", Path.cwd())).resolve()
        p = (repo_root / p).resolve()

    if mkdir:
        p.mkdir(parents=True, exist_ok=True)

    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path in config does not exist: {dotted_key} -> {p}")

    return p
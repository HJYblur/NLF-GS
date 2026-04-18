import os
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


_CONFIG: Dict[str, Any] = {}
_LOADED: bool = False


def _default_config_path() -> str:
    return os.environ.get("NLFGS_CONFIG", "configs/nlfgs_gpu.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    global _CONFIG, _LOADED
    if _LOADED and not path:
        return _CONFIG
    cfg_path = path or _default_config_path()
    try:
        cfg = OmegaConf.load(cfg_path)
        _CONFIG = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        _LOADED = True
    except Exception:
        _CONFIG = {}
        _LOADED = True
        # TODO[run-pipeline]: Raise or log a clear error when config cannot be loaded;
        #   and validate presence of required sections (data, train, backbone, avatar_template).
    return _CONFIG


def get_config() -> Dict[str, Any]:
    if not _LOADED:
        load_config()
    return _CONFIG


def get(path: str, default: Any = None) -> Any:
    cfg = get_config()
    cur: Any = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

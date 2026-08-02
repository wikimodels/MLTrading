"""Configuration loader supporting dual-bot architecture."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
import yaml

load_dotenv()

_ACTIVE_BOT: str = os.getenv("ACTIVE_BOT", "ml_swing_bot")


def set_active_bot(bot_name: str) -> None:
    """Sets the active bot whose settings.yaml should be loaded."""
    global _ACTIVE_BOT
    if _ACTIVE_BOT != bot_name:
        _ACTIVE_BOT = bot_name
        get_config.cache_clear()
        logger.debug(f"Active bot switched to: {_ACTIVE_BOT}")


def get_project_root() -> Path:
    """Returns the root directory of the repository (parent of shared/)."""
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def get_config(bot_name: Optional[str] = None) -> dict:
    """Loads settings.yaml for the active (or specified) bot."""
    target_bot = bot_name or _ACTIVE_BOT
    root_dir = get_project_root()
    config_path = root_dir / target_bot / "config" / "settings.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found for bot '{target_bot}': {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

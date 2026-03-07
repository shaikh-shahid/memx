import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_CONFIG = Path(__file__).parent.parent / "config.toml"
USER_CONFIG_DIR = Path.home() / ".memx"
USER_CONFIG = USER_CONFIG_DIR / "config.toml"
LEGACY_CONFIG_DIR = Path.home() / ".memex"
LEGACY_CONFIG = LEGACY_CONFIG_DIR / "config.toml"


@dataclass
class OllamaConfig:
    base_url: str
    embed_model: str
    extract_model: str
    timeout_seconds: int


@dataclass
class StorageConfig:
    db_path: Path


@dataclass
class MemoryConfig:
    extract_threshold_chars: int
    default_scope: str
    project_ttl_days: int
    embedding_dim: int = 768


@dataclass
class Config:
    ollama: OllamaConfig
    storage: StorageConfig
    memory: MemoryConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config() -> Config:
    with open(PACKAGE_CONFIG, "rb") as f:
        data = tomllib.load(f)

    # Best-effort migration from legacy memex directory on first memx run.
    if not USER_CONFIG_DIR.exists() and LEGACY_CONFIG_DIR.exists():
        try:
            shutil.copytree(LEGACY_CONFIG_DIR, USER_CONFIG_DIR, dirs_exist_ok=True)
        except OSError:
            pass

    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    if not USER_CONFIG.exists():
        if LEGACY_CONFIG.exists():
            shutil.copy(LEGACY_CONFIG, USER_CONFIG)
        else:
            shutil.copy(PACKAGE_CONFIG, USER_CONFIG)
    try:
        USER_CONFIG_DIR.chmod(0o700)
        USER_CONFIG.chmod(0o600)
    except OSError:
        # Best effort only; may fail on non-POSIX filesystems.
        pass

    if USER_CONFIG.exists():
        with open(USER_CONFIG, "rb") as f:
            user_data = tomllib.load(f)
        _deep_merge(data, user_data)

    memory_data = data["memory"]
    if "embedding_dim" not in memory_data:
        memory_data["embedding_dim"] = 768

    return Config(
        ollama=OllamaConfig(**data["ollama"]),
        storage=StorageConfig(db_path=Path(data["storage"]["db_path"]).expanduser()),
        memory=MemoryConfig(**memory_data),
    )

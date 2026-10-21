"""Settings from environment variables (`.env` is loaded by `uv run --env-file .env`)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# The one routing table. Platform hosts serve league/mastery/spectator; regional hosts serve match-v5.
PLATFORM_TO_REGIONAL: dict[str, str] = {
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}

# account-v1 has no 'sea' cluster; those platforms resolve accounts via 'asia'.
_ACCOUNT_REGIONAL: dict[str, str] = {"sea": "asia"}


class ConfigError(Exception):
    pass


class ApiKey:
    """The Riot API key. Its repr is masked so it never ends up in logs or tracebacks."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def __repr__(self) -> str:
        return "ApiKey('RGAPI-****')"

    __str__ = __repr__

    def reveal(self) -> str:
        """The raw key. Only the HTTP client's auth header calls this."""
        return self._value


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: ApiKey
    platform: str          # e.g. 'euw1'
    data_dir: Path

    @property
    def regional(self) -> str:
        return PLATFORM_TO_REGIONAL[self.platform]

    @property
    def account_regional(self) -> str:
        return _ACCOUNT_REGIONAL.get(self.regional, self.regional)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "oracle.sqlite3"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "riot.lock"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ, *, require_key: bool = True) -> Settings:
        platform = env.get("RIOT_PLATFORM", "").strip().lower()
        if platform not in PLATFORM_TO_REGIONAL:
            raise ConfigError(
                f"RIOT_PLATFORM must be one of {sorted(PLATFORM_TO_REGIONAL)}, got {platform!r}. "
                "Did you run with `uv run --env-file .env ...`?"
            )
        key = env.get("RIOT_API_KEY", "").strip()
        if require_key and not key.startswith("RGAPI-"):
            raise ConfigError("RIOT_API_KEY is missing or malformed (expected 'RGAPI-...'). Put it in .env.")
        data_dir = Path(env.get("ORACLE_DATA_DIR", "./data"))
        return cls(api_key=ApiKey(key), platform=platform, data_dir=data_dir)

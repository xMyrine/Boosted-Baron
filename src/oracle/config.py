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


DEFAULT_MODEL_DIR = "./training/artifacts/win_lr"


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


class DiscordToken:
    """The Discord bot token. Masked like `ApiKey`; only the Discord login calls `reveal`."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def __repr__(self) -> str:
        return "DiscordToken('****')"

    __str__ = __repr__

    def reveal(self) -> str:
        return self._value


@dataclass(frozen=True, slots=True)
class DiscordSettings:
    token: DiscordToken
    channel_id: int


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: ApiKey
    platform: str          # e.g. 'euw1'
    data_dir: Path
    model_dir: Path = Path(DEFAULT_MODEL_DIR)   # a folder with model.joblib + model_card.json
    discord: DiscordSettings | None = None      # only parsed (and required) by commands that talk to Discord

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
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        require_key: bool = True,
        require_model: bool = False,
        require_discord: bool = False,
    ) -> Settings:
        """Read settings from `env`, or by default from the process environment plus the nearest `.env`
        (searched upwards from the current directory, so commands work from any subfolder of the repo).
        Real environment variables win over `.env`. Relative ORACLE_DATA_DIR and ORACLE_MODEL_DIR are relative
        to the `.env`. The model folder and the Discord token and channel are checked only when required, so
        `oracle crawl` never fails over a bot setting."""
        base = Path.cwd()
        if env is None:
            dotenv = find_dotenv(base)
            if dotenv is not None:
                base = dotenv.parent
            env = {**(read_dotenv(dotenv) if dotenv else {}), **os.environ}
        platform = env.get("RIOT_PLATFORM", "").strip().lower()
        if platform not in PLATFORM_TO_REGIONAL:
            raise ConfigError(
                f"RIOT_PLATFORM must be one of {sorted(PLATFORM_TO_REGIONAL)}, got {platform!r}. "
                "Set it in .env (copy .env.example)."
            )
        key = env.get("RIOT_API_KEY", "").strip()
        if require_key and not key.startswith("RGAPI-"):
            raise ConfigError("RIOT_API_KEY is missing or malformed (expected 'RGAPI-...'). Put it in .env.")
        data_dir = Path(env.get("ORACLE_DATA_DIR", "./data"))
        model_dir = base / Path(env.get("ORACLE_MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR)
        if require_model:
            _check_model_dir(model_dir)
        discord = _parse_discord(env) if require_discord else None
        return cls(api_key=ApiKey(key), platform=platform, data_dir=base / data_dir, model_dir=model_dir,
                   discord=discord)


def _check_model_dir(model_dir: Path) -> None:
    missing = [name for name in ("model.joblib", "model_card.json") if not (model_dir / name).is_file()]
    if missing:
        raise ConfigError(
            f"ORACLE_MODEL_DIR ({model_dir}) has no {' and no '.join(missing)}. Train a model in training/ and save "
            "it with trainlib.save_model, or point ORACLE_MODEL_DIR at a saved model folder."
        )


def _parse_discord(env: Mapping[str, str]) -> DiscordSettings:
    token = env.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise ConfigError("DISCORD_BOT_TOKEN is missing. Create a bot in the Discord developer portal and put its "
                          "token in .env (see docs/wiki/Discord-Bot.md).")
    if token.count(".") != 2 or any(ch.isspace() for ch in token):
        raise ConfigError("DISCORD_BOT_TOKEN does not look like a bot token (three parts separated by dots). "
                          "Copy it from the Bot page of the application, not the application id or public key.")
    channel = env.get("DISCORD_CHANNEL_ID", "").strip()
    if not channel.isdigit() or int(channel) == 0:
        raise ConfigError("DISCORD_CHANNEL_ID must be the numeric id of the channel (enable Developer Mode in "
                          f"Discord, right-click the channel, Copy Channel ID), got {channel!r}.")
    return DiscordSettings(DiscordToken(token), int(channel))


def find_dotenv(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser: skips blank lines and comments, strips optional quotes."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values

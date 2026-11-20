"""Bot settings are parsed at the boundary: clear errors, masked secrets, and required only where used."""

from __future__ import annotations

import pytest

from oracle.config import ConfigError, DiscordToken, Settings

BASE = {"RIOT_PLATFORM": "eun1", "RIOT_API_KEY": "RGAPI-x"}
TOKEN = "MTIzNDU2.GabCDE.xyz_token_part"


def settings(tmp_path, **env) -> Settings:
    """Settings from a throwaway directory, so the repo's real .env is never read."""
    return Settings.from_env({**BASE, "ORACLE_DATA_DIR": str(tmp_path), **env}, require_key=False)


def test_discord_settings_are_parsed_when_required(tmp_path):
    s = Settings.from_env({**BASE, "DISCORD_BOT_TOKEN": TOKEN, "DISCORD_CHANNEL_ID": "1234567890"}, require_discord=True)
    assert s.discord is not None and s.discord.channel_id == 1234567890 and s.discord.token.reveal() == TOKEN


def test_commands_that_do_not_talk_to_discord_never_fail_over_its_settings():
    s = Settings.from_env({**BASE, "DISCORD_BOT_TOKEN": "garbage", "DISCORD_CHANNEL_ID": "nope"})
    assert s.discord is None


def test_the_token_never_appears_in_a_repr_or_a_traceback():
    s = Settings.from_env({**BASE, "DISCORD_BOT_TOKEN": TOKEN, "DISCORD_CHANNEL_ID": "5"}, require_discord=True)
    assert TOKEN not in repr(s) and TOKEN not in str(s.discord) and TOKEN not in f"{s.discord.token}"
    assert repr(DiscordToken("a.b.c")) == "DiscordToken('****')"


@pytest.mark.parametrize("env, message", [
    ({}, "DISCORD_BOT_TOKEN is missing"),
    ({"DISCORD_BOT_TOKEN": "just-an-application-id", "DISCORD_CHANNEL_ID": "5"}, "does not look like a bot token"),
    ({"DISCORD_BOT_TOKEN": TOKEN}, "DISCORD_CHANNEL_ID must be the numeric id"),
    ({"DISCORD_BOT_TOKEN": TOKEN, "DISCORD_CHANNEL_ID": "#general"}, "DISCORD_CHANNEL_ID must be the numeric id"),
])
def test_missing_or_malformed_discord_settings_say_what_to_fix(env, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({**BASE, **env}, require_discord=True)


def test_a_bad_token_error_does_not_echo_the_token():
    with pytest.raises(ConfigError) as caught:
        Settings.from_env({**BASE, "DISCORD_BOT_TOKEN": "secretvalue", "DISCORD_CHANNEL_ID": "5"}, require_discord=True)
    assert "secretvalue" not in str(caught.value)


def test_the_model_dir_defaults_to_win_lr_and_can_be_overridden(tmp_path):
    assert settings(tmp_path).model_dir.parts[-3:] == ("training", "artifacts", "win_lr")
    assert settings(tmp_path, ORACLE_MODEL_DIR="/models/other").model_dir.as_posix() == "/models/other"
    assert settings(tmp_path, ORACLE_MODEL_DIR="  ").model_dir.name == "win_lr"        # an empty placeholder in .env


def test_a_model_dir_without_the_two_files_is_refused_when_required(tmp_path):
    env = {**BASE, "ORACLE_MODEL_DIR": str(tmp_path)}
    with pytest.raises(ConfigError, match="has no model.joblib and no model_card.json"):
        Settings.from_env(env, require_key=False, require_model=True)
    (tmp_path / "model.joblib").write_bytes(b"x")
    with pytest.raises(ConfigError, match="has no model_card.json"):
        Settings.from_env(env, require_key=False, require_model=True)
    (tmp_path / "model_card.json").write_text("{}")
    assert Settings.from_env(env, require_key=False, require_model=True).model_dir == tmp_path

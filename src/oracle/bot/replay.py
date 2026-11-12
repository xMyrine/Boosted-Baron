"""Replay a finished game through the exact pipeline a live game takes, and show the real result beside it.

The stored game is turned back into the spectator payload it would have been at its start and goes through
`Pregame.from_spectator`, `compute` and `foresee`, like the service does. No network, and no database writes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from oracle.bot.champions import ChampionNames
from oracle.bot.forecast import compute, foresee
from oracle.bot.predictor import Predictor
from oracle.bot.prophecy import Message, Prophecy, render
from oracle.config import Settings
from oracle.features import FeatureParams, Pregame, Side
from oracle.mirror import (
    Mirror,
    StoredGame,
    StoredPlayer,
    champion_names,
    newest_complete_anchor,
    stored_game,
    tracked_names,
)


@dataclass(frozen=True, slots=True)
class Replay:
    game: StoredGame
    prophecy: Prophecy
    message: Message          # the rendered prophecy plus a line with the real result


def replay(settings: Settings, match_id: str | None, seat: int | None) -> Replay:
    """`seat` is the participant number 1..10 who plays the subject. Default: the first tracked player in the game."""
    if not settings.db_path.is_file():
        raise LookupError(f"no database at {settings.db_path}. Run `oracle crawl` first.")
    predictor = Predictor.load(settings.model_dir)
    params = FeatureParams()
    con = Mirror.read_only(settings.db_path)
    try:
        match_id = match_id or newest_complete_anchor(con, queue_id=params.queue_id, history_n=params.history_n)
        if match_id is None:
            raise LookupError("no stored game has a complete history yet. Run `oracle crawl` first.")
        game = stored_game(con, match_id)
        if game is None:
            raise LookupError(f"{match_id} is not stored. `oracle status` shows what the database holds.")
        names, tracked = ChampionNames(champion_names(con)), tracked_names(con)
    finally:
        con.close()

    subject = _subject(game, seat, tracked)
    pregame = Pregame.from_spectator(game.spectator_payload(), now_ms=game.game_start_ms)
    started = time.monotonic()
    features, prediction = compute(settings.db_path, pregame, predictor, params)
    prophecy = foresee(pregame, subject.puuid, tracked.get(subject.puuid, "Summoner"), features, prediction, names,
                       model_id=predictor.model_id, prepared_in_s=time.monotonic() - started)
    message = render(prophecy)
    message = replace(message, lines=(*message.lines, _result_line(game, prophecy)))
    return Replay(game, prophecy, message)


def _subject(game: StoredGame, seat: int | None, tracked: dict[str, str]) -> StoredPlayer:
    if seat is not None:
        if not 1 <= seat <= len(game.players):
            raise ValueError(f"--seat must be between 1 and {len(game.players)}, got {seat}")
        return game.players[seat - 1]
    return next((p for p in game.players if p.puuid in tracked), game.players[0])


def _result_line(game: StoredGame, prophecy: Prophecy) -> str:
    winner = {100: Side.BLUE, 200: Side.RED}.get(game.winning_team_id or 0)
    pct = round(100 * (prophecy.p_subject_win or 0.0))
    if winner is None:
        return f"Replay of {game.match_id}: the stored game has no result."
    outcome = "won" if winner is prophecy.subject_side else "lost"
    return (f"Replay of {game.match_id}, a finished game. Real result: {winner.name.lower()} won, so "
            f"{prophecy.subject_name}'s side {outcome}. The oracle said {pct}%.")

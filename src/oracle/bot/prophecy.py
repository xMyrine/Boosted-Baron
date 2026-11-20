"""The numbers of one game, and how they become a message. Pure: no I/O, no Discord, no scikit-learn.

    Prophecy(game_key, subject, ..., outcome=Foreseen(p_blue, drivers, seats) | Clouded(reason))
    render(prophecy) -> Message        # a pure function of the numbers; an LLM renderer can replace it

`Message` knows nothing about discord.py: the announcer converts it. Every probability, driver and quip in the
text comes from the `Prophecy`, so the stored JSON stays the source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, assert_never

from oracle.bot import phrasebook
from oracle.bot.phrasebook import Family
from oracle.features.pregame import Side

ROLE_ORDER = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
SCHEMA = 1   # version of the stored JSON, so a later change to the shape can still read old rows


@dataclass(frozen=True, slots=True)
class Driver:
    """One input of the model and how far it moved the SUBJECT's win probability (positive = helps)."""

    column: str
    family: Family
    label: str
    points: float   # percentage points: win probability with the driver minus without it


@dataclass(frozen=True, slots=True)
class SeatView:
    """What the message may say about one player. No PUUID: strangers are described by role and champion."""

    side: Side
    role: str | None            # inferred lane: TOP | JUNGLE | MIDDLE | BOTTOM | UTILITY
    champion: str
    is_subject: bool
    n_games: int                # usable history games (0..history_n)
    streak: int                 # signed run from the most recent game: +wins / -losses
    champ_games: int            # history games on this champion
    main_role: str | None       # most played role in the history
    off_role: bool | None       # None = no history to judge
    history_complete: bool


@dataclass(frozen=True, slots=True)
class Foreseen:
    p_blue: float
    drivers: tuple[Driver, ...]   # strongest first, signed for the subject; empty if the model cannot explain itself
    seats: tuple[SeatView, ...]   # 5 blue then 5 red

    @property
    def histories_missing(self) -> int:
        return sum(not seat.history_complete for seat in self.seats)

    @property
    def complete(self) -> bool:
        return self.histories_missing == 0


class Reason(StrEnum):
    PREDICTION_FAILED = "prediction_failed"   # building features or running the model raised
    RIOT_REFUSED = "riot_refused"             # 401/403 while preparing: no data is coming


@dataclass(frozen=True, slots=True)
class Clouded:
    reason: Reason


@dataclass(frozen=True, slots=True)
class Prophecy:
    game_key: str            # 'EUN1_4017659773': also the match id once the game is over
    subject: str             # PUUID of the tracked player whose spectator check found the game
    subject_name: str        # the tracked player's game name: friends who opted in, shown in the title
    subject_side: Side
    model_id: str
    prepared_in_s: float
    outcome: Foreseen | Clouded

    @property
    def p_subject_win(self) -> float | None:
        match self.outcome:
            case Foreseen(p_blue=p):
                return p if self.subject_side is Side.BLUE else 1.0 - p
            case Clouded():
                return None
            case _ as never:
                assert_never(never)

    def to_json(self) -> str:
        outcome: dict[str, Any]
        match self.outcome:
            case Foreseen(p_blue=p, drivers=drivers, seats=seats):
                outcome = {
                    "kind": "foreseen", "p_blue": p,
                    "drivers": [{"column": d.column, "family": d.family.value, "label": d.label, "points": d.points}
                                for d in drivers],
                    "seats": [{"side": int(s.side), "role": s.role, "champion": s.champion, "is_subject": s.is_subject,
                               "n_games": s.n_games, "streak": s.streak, "champ_games": s.champ_games,
                               "main_role": s.main_role, "off_role": s.off_role,
                               "history_complete": s.history_complete} for s in seats],
                }
            case Clouded(reason=reason):
                outcome = {"kind": "clouded", "reason": reason.value}
            case _ as never:
                assert_never(never)
        return json.dumps({
            "v": SCHEMA, "game_key": self.game_key, "subject": self.subject, "subject_name": self.subject_name,
            "subject_side": int(self.subject_side), "model_id": self.model_id, "prepared_in_s": self.prepared_in_s,
            "outcome": outcome,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Prophecy:
        raw = json.loads(text)
        if raw.get("v") != SCHEMA:
            raise ValueError(f"prophecy JSON version {raw.get('v')!r} is not {SCHEMA}")
        o = raw["outcome"]
        outcome: Foreseen | Clouded
        match o["kind"]:
            case "foreseen":
                outcome = Foreseen(
                    o["p_blue"],
                    tuple(Driver(d["column"], Family(d["family"]), d["label"], d["points"]) for d in o["drivers"]),
                    tuple(SeatView(Side(s["side"]), s["role"], s["champion"], s["is_subject"], s["n_games"],
                                   s["streak"], s["champ_games"], s["main_role"], s["off_role"],
                                   s["history_complete"]) for s in o["seats"]),
                )
            case "clouded":
                outcome = Clouded(Reason(o["reason"]))
            case kind:
                raise ValueError(f"unknown outcome kind {kind!r}")
        return cls(raw["game_key"], raw["subject"], raw["subject_name"], Side(raw["subject_side"]), raw["model_id"],
                   raw["prepared_in_s"], outcome)


# ---------------------------------------------------------------- the rendered message

@dataclass(frozen=True, slots=True)
class Field:
    name: str
    value: str
    inline: bool = True


@dataclass(frozen=True, slots=True)
class Message:
    """Everything an announcer needs, in no particular chat platform's types."""

    title: str
    lines: tuple[str, ...]      # body paragraphs; **bold** is the only markup
    fields: tuple[Field, ...]
    footer: str
    colour: int                 # 0xRRGGBB

    def as_text(self) -> str:
        parts = [self.title, "", *self.lines]
        for f in self.fields:
            parts += ["", f"{f.name}:", *("  " + row for row in f.value.splitlines())]
        parts += ["", self.footer]
        return "\n".join(parts).replace("**", "")


COLOURS = {"good": 0x2ECC71, "even": 0x95A5A6, "bad": 0xE67E22, "doom": 0xE74C3C, "flagged": 0xF1C40F, "clouded": 0x607D8B}


def render(p: Prophecy) -> Message:
    match p.outcome:
        case Foreseen() as f:
            return _render_foreseen(p, f)
        case Clouded(reason=reason):
            return Message(
                title=f"The vision is clouded for {p.subject_name}",
                lines=(phrasebook.clouded_line(reason, p.game_key),),
                fields=(),
                footer=f"{_model_name(p.model_id)} · {p.game_key}",
                colour=COLOURS["clouded"],
            )
        case _ as never:
            assert_never(never)


def _render_foreseen(p: Prophecy, f: Foreseen) -> Message:
    p_win = p.p_subject_win
    assert p_win is not None
    facts = phrasebook.Facts.of(p.game_key, p.subject_name, p_win, f)
    lines = [phrasebook.opening(facts), *phrasebook.driver_lines(facts), *phrasebook.quips(facts)]
    if not f.complete:
        lines.append(phrasebook.flagged_line(facts, f.histories_missing))
    subject = next((s for s in f.seats if s.is_subject), None)
    title = f"Prophecy for {p.subject_name}"
    if subject is not None:
        title += f": {phrasebook.role_name(subject.role, short=True)} {subject.champion}"
    return Message(
        title=title,
        lines=tuple(lines),
        fields=tuple(_side_field(f, side) for side in (Side.BLUE, Side.RED)),
        footer=(f"{_model_name(p.model_id)} · {10 - f.histories_missing}/10 histories read · "
                f"{p.prepared_in_s:.0f}s · {p.game_key}"),
        colour=COLOURS["flagged"] if not f.complete else COLOURS[facts.band.colour],
    )


def _side_field(f: Foreseen, side: Side) -> Field:
    seats = sorted((s for s in f.seats if s.side is side),
                   key=lambda s: ROLE_ORDER.index(s.role) if s.role in ROLE_ORDER else len(ROLE_ORDER))
    rows = [f"{phrasebook.role_name(s.role, short=True)} " + (f"**{s.champion}**" if s.is_subject else s.champion)
            for s in seats]
    return Field(f"{side.name.title()} side (roles inferred)", "\n".join(rows))


def _model_name(model_id: str) -> str:
    return model_id.split(" ", 1)[0]

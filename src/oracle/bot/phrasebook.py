"""The oracle's voice as tables: what to say for which fact. Pure data and tiny lookups, no I/O.

Facts come from the numbers (probability band, strongest driver family, streaks, off-role, first-time champion).
Each fact kind owns a row of wording variants, and a hash of the game key picks the variant, so the same game
always reads the same and tests can assert literals. Add a joke by adding a row, not a branch.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oracle.bot.prophecy import Foreseen, Reason, SeatView
    from oracle.features.pregame import Side


class Family(StrEnum):
    """Which part of the feature catalogue a model input comes from (the five `seat.py` families)."""

    FORM = "form"
    PERFORMANCE = "performance"
    CHAMPION = "champion"
    ROLE = "role"
    RANK = "rank"
    SIDE = "side"
    OTHER = "other"


# Every numeric seat feature in `oracle.features.FEATURE_DOCS` needs a row; a test fails when a feature is added
# without one. Wording is neutral about direction: the sign of the driver says whether it helps.
FEATURE_LABELS: dict[str, tuple[Family, str]] = {
    "n_games": (Family.FORM, "games played recently"),
    "wr_shrunk": (Family.FORM, "win rate"),
    "wr_recent": (Family.FORM, "win rate over the last five games"),
    "streak": (Family.FORM, "win or loss streak"),
    "session_games": (Family.FORM, "games played today"),
    "session_losses": (Family.FORM, "losses today"),
    "hours_since_last": (Family.FORM, "time since the last game"),
    "kda": (Family.PERFORMANCE, "KDA"),
    "kills_pm": (Family.PERFORMANCE, "kills per minute"),
    "deaths_pm": (Family.PERFORMANCE, "deaths per minute"),
    "assists_pm": (Family.PERFORMANCE, "assists per minute"),
    "deaths_max": (Family.PERFORMANCE, "worst death count in one game"),
    "dmg_pm": (Family.PERFORMANCE, "damage per minute"),
    "dmg_taken_pm": (Family.PERFORMANCE, "damage taken per minute"),
    "gold_pm": (Family.PERFORMANCE, "gold per minute"),
    "cs_pm": (Family.PERFORMANCE, "CS per minute"),
    "vision_pm": (Family.PERFORMANCE, "vision per minute"),
    "kp": (Family.PERFORMANCE, "kill participation"),
    "dmg_share": (Family.PERFORMANCE, "share of team damage"),
    "champ_games": (Family.CHAMPION, "games on the picked champion"),
    "champ_wr_shrunk": (Family.CHAMPION, "win rate on the picked champion"),
    "days_since_champ": (Family.CHAMPION, "days since last playing the champion"),
    "champ_meta_wr": (Family.CHAMPION, "champion win rate in the current meta"),
    "champ_meta_games": (Family.CHAMPION, "champion popularity"),
    "role_share": (Family.ROLE, "share of games in the role"),
    "off_role": (Family.ROLE, "off-role rate"),
    "role_conf": (Family.ROLE, "how well the champion fits the lane"),
    "rank_score": (Family.RANK, "rank"),
    "ranked_games": (Family.RANK, "ranked experience"),
    "ranked_wr_shrunk": (Family.RANK, "season win rate"),
    "mastery_points_log": (Family.RANK, "mastery points"),
    "mastery_level": (Family.RANK, "mastery level"),
}

ROLE_NOUN = {"TOP": "top laner", "JUNGLE": "jungler", "MIDDLE": "mid laner", "BOTTOM": "bot laner", "UTILITY": "support"}
ROLE_SHORT = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "Bot", "UTILITY": "Support"}

_SCOPES = (
    ("d_mean_", "team average {}"),
    ("d_min_", "lowest {} on each side"),
    ("d_max_", "highest {} on each side"),
    ("ctx_", "all-player average {}"),
)


def role_name(role: str | None, *, short: bool = False) -> str:
    table = ROLE_SHORT if short else ROLE_NOUN
    return table.get(role or "", "player")


def describe_column(column: str) -> tuple[Family, str]:
    """A model input column ('d_mean_wr_shrunk', 'lane_MIDDLE_kda', 'missingindicator_...') -> family and label."""
    if column == "intercept":
        return Family.SIDE, "the side of the map"
    if column.startswith("missingindicator_"):
        family, label = describe_column(column.removeprefix("missingindicator_"))
        return family, f"missing data for the {label}"
    if column.startswith("lane_"):
        _, role, feature = column.split("_", 2)
        if feature in FEATURE_LABELS:
            family, label = FEATURE_LABELS[feature]
            return family, f"{role_name(role, short=True).lower()} lane matchup ({label})"
    for prefix, frame in _SCOPES:
        feature = column.removeprefix(prefix)
        if column.startswith(prefix) and feature in FEATURE_LABELS:
            family, label = FEATURE_LABELS[feature]
            return family, frame.format(label)
    return Family.OTHER, column


# ---------------------------------------------------------------- facts

@dataclass(frozen=True, slots=True)
class Band:
    name: str
    floor: float    # lowest P(subject wins) that belongs to this band
    colour: str     # key into prophecy.COLOURS


BANDS = (
    Band("doomed", 0.0, "doom"),
    Band("underdog", 0.40, "bad"),
    Band("coinflip", 0.46, "even"),
    Band("favoured", 0.54, "good"),
    Band("dominant", 0.60, "good"),
)


def band_of(p_win: float) -> Band:
    return [b for b in BANDS if p_win >= b.floor][-1]


@dataclass(frozen=True, slots=True)
class Facts:
    key: str
    name: str
    p_win: float
    band: Band
    foreseen: Foreseen

    @classmethod
    def of(cls, key: str, name: str, p_win: float, foreseen: Foreseen) -> Facts:
        return cls(key, name, p_win, band_of(p_win), foreseen)

    @property
    def pct(self) -> int:
        return round(self.p_win * 100)


def pick(options: tuple[str, ...], key: str, salt: str) -> str:
    """Deterministic variant: the same game key and slot always give the same wording."""
    return options[zlib.crc32(f"{key}|{salt}".encode()) % len(options)]


# ---------------------------------------------------------------- wording tables

OPENINGS: dict[str, tuple[str, ...]] = {
    "dominant": (
        "The stars align for {name}: **{pct}%** to win. Try not to throw it.",
        "**{pct}%** for {name}. The oracle has seen worse omens in a pentakill.",
    ),
    "favoured": (
        "The omens lean {name}'s way: **{pct}%** to win.",
        "**{pct}%** for {name}. Favoured, which is not the same as safe.",
    ),
    "coinflip": (
        "A coin flip dressed as destiny: **{pct}%** for {name}.",
        "**{pct}%** for {name}. The oracle shrugs, and the shrug is statistically sound.",
    ),
    "underdog": (
        "The omens frown: **{pct}%** for {name}. Not hopeless, merely unkind.",
        "Only **{pct}%** for {name}. Win it anyway and the oracle will pretend it never doubted.",
    ),
    "doomed": (
        "Brutal, but the numbers do not negotiate: **{pct}%** for {name}.",
        "**{pct}%** for {name}. The oracle recommends a good ward placement and low expectations.",
    ),
}

# Strongest driver: (family, does it help the subject) -> frames with {name}, {label}, {pts}.
DRIVER_LINES: dict[tuple[Family, bool], tuple[str, ...]] = {
    (Family.FORM, True): ("Recent form is the loudest voice: the {label} adds {pts} points for {name}.",),
    (Family.FORM, False): ("Recent form speaks against {name}: the {label} costs {pts} points.",),
    (Family.PERFORMANCE, True): ("The ledger of past games favours {name}: the {label} adds {pts} points.",),
    (Family.PERFORMANCE, False): ("The ledger of past games disagrees: the {label} costs {name} {pts} points.",),
    (Family.CHAMPION, True): ("The champion picks please the oracle: the {label} adds {pts} points.",),
    (Family.CHAMPION, False): ("The champion picks trouble the oracle: the {label} costs {pts} points.",),
    (Family.ROLE, True): ("The lanes sort themselves in {name}'s favour: the {label} adds {pts} points.",),
    (Family.ROLE, False): ("The lanes are muddled against {name}: the {label} costs {pts} points.",),
    (Family.RANK, True): ("Rank and mastery lean {name}'s way: the {label} adds {pts} points.",),
    (Family.RANK, False): ("Rank and mastery lean the other way: the {label} costs {name} {pts} points.",),
    (Family.SIDE, True): ("The map itself helps {name}: {label} adds {pts} points.",),
    (Family.SIDE, False): ("The map itself works against {name}: {label} costs {pts} points.",),
    (Family.OTHER, True): ("A quiet signal helps {name}: {label} adds {pts} points.",),
    (Family.OTHER, False): ("A quiet signal hurts {name}: {label} costs {pts} points.",),
}
CONTRAST_LINES = {
    True: ("Against that, the {label} costs {pts} points.",),
    False: ("In its favour, the {label} adds {pts} points.",),
}
FAINT = ("No single signal moves the odds by more than {pts} points. The oracle is mostly guessing.",)
FAINT_BELOW_POINTS = 0.5
CONTRAST_FROM_POINTS = 0.5

FLAGGED = (
    "The vision is clouded: only {read} of 10 histories could be read in time. Treat this number as a rumour.",
    "Only {read} of 10 histories reached the oracle before the deadline. This number is a rumour with decimals.",
)

CLOUDED: dict[str, str] = {
    "prediction_failed": "Something broke between the data and the prophecy for {key}. The omen is lost.",
    "riot_refused": "Riot refused the key while the histories for {key} were read. Check RIOT_API_KEY and wait for the next game.",
}


def opening(facts: Facts) -> str:
    return pick(OPENINGS[facts.band.name], facts.key, "opening").format(name=facts.name, pct=facts.pct)


def driver_lines(facts: Facts) -> list[str]:
    drivers = facts.foreseen.drivers
    if not drivers:
        return []
    lead = drivers[0]
    if abs(lead.points) < FAINT_BELOW_POINTS:
        return [pick(FAINT, facts.key, "faint").format(pts=f"{FAINT_BELOW_POINTS:.1f}")]
    helps = lead.points > 0
    lines = [pick(DRIVER_LINES[(lead.family, helps)], facts.key, "driver").format(
        name=facts.name, label=lead.label, pts=f"{abs(lead.points):.1f}")]
    contrast = next((d for d in drivers[1:] if (d.points > 0) != helps and abs(d.points) >= CONTRAST_FROM_POINTS), None)
    if contrast is not None:
        lines.append(pick(CONTRAST_LINES[helps], facts.key, "contrast").format(
            label=contrast.label, pts=f"{abs(contrast.points):.1f}"))
    return lines


def flagged_line(facts: Facts, missing: int) -> str:
    return pick(FLAGGED, facts.key, "flagged").format(read=10 - missing)


def clouded_line(reason: Reason, key: str) -> str:
    return CLOUDED[reason.value].format(key=key)


# ---------------------------------------------------------------- quips: facts about single players

class Relation(StrEnum):
    SUBJECT = "subject"
    ALLY = "ally"
    ENEMY = "enemy"


@dataclass(frozen=True, slots=True)
class Quip:
    """One kind of joke: how to notice it on a seat (a strength, or None) and the wording per relation."""

    kind: str
    detect: Callable[[SeatView], int | None]
    lines: dict[Relation, tuple[str, ...]]


QUIPS = (
    Quip("losing_streak", lambda s: -s.streak if s.streak <= -3 else None, {
        Relation.SUBJECT: ("{who} walks in on a {n}-game losing streak. The oracle does not say tilt. The numbers do.",),
        Relation.ALLY: ("{who} has lost {n} games in a row. Pray for them.",),
        Relation.ENEMY: ("{who} has lost {n} games in a row. Mercy is not on the menu.",),
    }),
    Quip("winning_streak", lambda s: s.streak if s.streak >= 4 else None, {
        Relation.SUBJECT: ("{who} arrives on a {n}-game winning streak. Confidence is allowed. Complacency is not.",),
        Relation.ALLY: ("{who} has won {n} in a row. Stand near them.",),
        Relation.ENEMY: ("{who} has won {n} in a row. Someone has to stop them, and it will not be the oracle.",),
    }),
    Quip("off_role", lambda s: 3 if s.off_role else None, {
        Relation.SUBJECT: ("{who} is off-role: the history says {main}, today says otherwise.",),
        Relation.ALLY: ("{who} is off-role: their history says {main}. Brave.",),
        Relation.ENEMY: ("{who} is off-role: their history says {main}. The enemy is improvising.",),
    }),
    Quip("first_time", lambda s: 3 if s.champ_games == 0 and s.n_games >= 5 else None, {
        Relation.SUBJECT: ("{who} has not played this champion in the last {games} games. A bold experiment.",),
        Relation.ALLY: ("{who} is on a champion they have not touched in {games} games.",),
        Relation.ENEMY: ("{who} is on a champion they have not touched in {games} games. Let them learn.",),
    }),
)
RELATION_BONUS = {Relation.SUBJECT: 2, Relation.ALLY: 1, Relation.ENEMY: 0}
MAX_QUIPS = 2


def _relation(seat: SeatView, subject_side: Side) -> Relation:
    if seat.is_subject:
        return Relation.SUBJECT
    return Relation.ALLY if seat.side == subject_side else Relation.ENEMY


def _who(seat: SeatView, relation: Relation, name: str) -> str:
    role = role_name(seat.role)
    match relation:
        case Relation.SUBJECT:
            return f"{name} on {seat.champion}"
        case Relation.ALLY:
            return f"{name}'s {role} on {seat.champion}"
        case Relation.ENEMY:
            return f"the enemy {role} on {seat.champion}"


def quips(facts: Facts) -> list[str]:
    """The most striking facts about single players, subject first, at most one per kind."""
    seats = facts.foreseen.seats
    subject = next((s for s in seats if s.is_subject), None)
    if subject is None:
        return []
    found = []
    for quip in QUIPS:
        for seat in seats:
            strength = quip.detect(seat)
            if strength is not None:
                relation = _relation(seat, subject.side)
                found.append((strength + RELATION_BONUS[relation], quip, seat, relation, strength))
    found.sort(key=lambda item: -item[0])   # stable: earlier kinds and seats win ties
    lines: list[str] = []
    used: set[str] = set()
    for _, quip, seat, relation, strength in found:
        if quip.kind in used or len(lines) == MAX_QUIPS:
            continue
        used.add(quip.kind)
        template = pick(quip.lines[relation], facts.key, f"quip:{quip.kind}")
        line = template.format(who=_who(seat, relation, facts.name), n=strength, games=seat.n_games,
                               main=role_name(seat.main_role))
        lines.append(line[0].upper() + line[1:])
    return lines

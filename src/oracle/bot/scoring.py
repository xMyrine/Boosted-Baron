"""How good were the stored prophecies? Log loss and Brier score against the constant 0.5 baseline. Pure maths.

Every prophecy is stored before its game ends, so this is the one evaluation that cannot leak. Log loss
punishes a confident wrong answer hard; Brier is the mean squared error of the probability. A model that always
says 50% scores 0.6931 and 0.25, so those are the numbers to beat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from oracle.mirror import ProphecyCounts, ScoredProphecy

EPS = 1e-6
COIN_FLIP_LOG_LOSS = math.log(2)
COIN_FLIP_BRIER = 0.25
FEW_GAMES = 100


@dataclass(frozen=True, slots=True)
class Score:
    n: int
    log_loss: float
    brier: float


def score(rows: Sequence[ScoredProphecy]) -> Score | None:
    if not rows:
        return None
    ll = br = 0.0
    for row in rows:
        p = min(max(row.p_blue, EPS), 1 - EPS)
        y = 1.0 if row.blue_won else 0.0
        ll -= y * math.log(p) + (1 - y) * math.log(1 - p)
        br += (p - y) ** 2
    return Score(len(rows), ll / len(rows), br / len(rows))


def report(counts: ProphecyCounts, scored: Sequence[ScoredProphecy]) -> str:
    lines = [f"prophecies  {counts.pending} pending, {counts.posted} posted, {counts.failed} failed"]
    overall = score(scored)
    if overall is None:
        waiting = counts.with_number
        lines.append(f"scored      none yet ({waiting} waiting for their game to be fetched)" if waiting
                     else "scored      none yet (no prophecy has been made)")
        return "\n".join(lines)
    lines.append(f"scored      {overall.n} finished games, {counts.with_number - overall.n} still waiting for the "
                 "crawler to fetch them")
    lines.append(f"{'':12}{'log loss':>9}{'brier':>8}")
    lines.append(f"{'oracle':12}{overall.log_loss:>9.4f}{overall.brier:>8.4f}")
    complete = score([r for r in scored if r.complete])
    if complete is not None and complete.n != overall.n:
        lines.append(f"{'10/10 reads':12}{complete.log_loss:>9.4f}{complete.brier:>8.4f}   n={complete.n}")
    lines.append(f"{'coin flip':12}{COIN_FLIP_LOG_LOSS:>9.4f}{COIN_FLIP_BRIER:>8.4f}")
    if overall.n < FEW_GAMES:
        lines.append(f"n is {overall.n}: differences under about 0.02 are noise below a few hundred games.")
    return "\n".join(lines)

"""Champion id -> readable name. Built from the stored match JSON (no network); unknown ids get a neutral label."""

from __future__ import annotations

import re
from collections.abc import Mapping

# Data Dragon ids that a plain camel-case split would get wrong.
SPECIAL = {
    "MonkeyKing": "Wukong", "FiddleSticks": "Fiddlesticks", "Leblanc": "LeBlanc", "Chogath": "Cho'Gath",
    "Khazix": "Kha'Zix", "Velkoz": "Vel'Koz", "Kaisa": "Kai'Sa", "KogMaw": "Kog'Maw", "RekSai": "Rek'Sai",
    "KSante": "K'Sante", "Belveth": "Bel'Veth", "DrMundo": "Dr. Mundo", "Nunu": "Nunu & Willump",
    "Renata": "Renata Glasc",
}


def readable(ddragon_id: str) -> str:
    return SPECIAL.get(ddragon_id) or re.sub(r"(?<=[a-z])(?=[A-Z])", " ", ddragon_id)


class ChampionNames:
    def __init__(self, ddragon_ids: Mapping[int, str]) -> None:
        self._names = {champion_id: readable(name) for champion_id, name in ddragon_ids.items()}

    def __getitem__(self, champion_id: int) -> str:
        return self._names.get(champion_id) or f"Champion #{champion_id}"

    def __len__(self) -> int:
        return len(self._names)

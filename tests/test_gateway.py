import asyncio
import random

import httpx
import pytest

from oracle.config import ApiKey, Settings
from oracle.ids import MatchId, Puuid
from oracle.riot import GONE, AuthError, BadRequestError, KeyInUse, RiotGateway, TransientError, classify
from oracle.riot.gateway import Fatal, NotFound, Ok, Throttled, Transient

LIMITS = {"X-App-Rate-Limit": "100:1", "X-App-Rate-Limit-Count": "1:1",
          "X-Method-Rate-Limit": "100:1", "X-Method-Rate-Limit-Count": "1:1"}


class NoJitter(random.Random):
    def uniform(self, a: float, b: float) -> float:
        return 0.0


def gateway(tmp_path, handler) -> RiotGateway:
    settings = Settings(ApiKey("RGAPI-test"), "euw1", tmp_path)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"X-Riot-Token": "RGAPI-test"})

    async def no_sleep(_: float) -> None:
        await asyncio.sleep(0)

    return RiotGateway(settings, client, None, sleep=no_sleep, lane_options={"pad_s": 0.0, "rng": NoJitter()})


def test_classify():
    assert classify(200, {}) == Ok()
    assert classify(404, {}) == NotFound()
    assert classify(429, {"Retry-After": "7", "X-Rate-Limit-Type": "application"}) == Throttled("application", 7.0)
    assert classify(429, {}) == Throttled(None, None)
    assert classify(503, {}) == Transient("http 503")
    assert classify(None, {}) == Transient("network")
    assert classify(403, {}) == Fatal(403)


def test_routes_to_regional_host_with_key_header(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=["EUW1_1"], headers=LIMITS)

    async def scenario():
        async with gateway(tmp_path, handler) as gw:
            return await gw.background().match_ids(Puuid("p"), queue=420, end_ms=1_700_000_000_999)

    assert asyncio.run(scenario()) == ["EUW1_1"]
    assert seen[0].url.host == "europe.api.riotgames.com"
    assert seen[0].url.params["endTime"] == "1700000000"           # ms -> s at the boundary
    assert seen[0].headers["X-Riot-Token"] == "RGAPI-test"


def test_application_429_is_retried_transparently(tmp_path):
    responses = [httpx.Response(429, headers={**LIMITS, "Retry-After": "0.05", "X-Rate-Limit-Type": "application"}),
                 httpx.Response(200, json={"ok": 1}, headers=LIMITS)]

    async def scenario():
        async with gateway(tmp_path, lambda _: responses.pop(0)) as gw:
            doc = await gw.background().match(MatchId("EUW1_1"))
            return doc, gw.violations

    doc, violations = asyncio.run(scenario())
    assert doc.json() == {"ok": 1} and violations == 1


def test_404_is_data_only_where_the_endpoint_says_so(tmp_path):
    async def scenario():
        async with gateway(tmp_path, lambda _: httpx.Response(404, headers=LIMITS)) as gw:
            assert await gw.background().match(MatchId("EUW1_1")) is GONE
            assert await gw.background().active_game(Puuid("p")) is None
            with pytest.raises(BadRequestError):
                await gw.background().match_ids(Puuid("p"), queue=420)

    asyncio.run(scenario())


def test_401_trips_the_gateway_for_every_later_call(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, headers=LIMITS)

    async def scenario():
        async with gateway(tmp_path, handler) as gw:
            for _ in range(2):
                with pytest.raises(AuthError):
                    await gw.background().match(MatchId("EUW1_1"))

    asyncio.run(scenario())
    assert len(calls) == 1


def test_5xx_retries_then_gives_up(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, headers=LIMITS)

    async def scenario():
        async with gateway(tmp_path, handler) as gw:
            with pytest.raises(TransientError):
                await gw.background().match(MatchId("EUW1_1"))

    asyncio.run(scenario())
    assert len(calls) == 4          # 1 + 3 background retries


def test_identical_concurrent_requests_share_one_call(tmp_path):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"m": 1}, headers=LIMITS)

    async def scenario():
        async with gateway(tmp_path, handler) as gw:
            api = gw.background()
            return await asyncio.gather(api.match(MatchId("EUW1_1")), api.match(MatchId("EUW1_1")))

    first, second = asyncio.run(scenario())
    assert first == second and len(calls) == 1


def test_one_process_owns_the_key(tmp_path):
    settings = Settings(ApiKey("RGAPI-test"), "euw1", tmp_path)

    async def scenario():
        async with RiotGateway.open(settings):
            with pytest.raises(KeyInUse):
                RiotGateway.open(settings)
        async with RiotGateway.open(settings):    # released on close
            pass

    asyncio.run(scenario())


def test_api_key_never_appears_in_repr():
    assert "secret" not in repr(ApiKey("RGAPI-secret"))
    assert "secret" not in repr(Settings(ApiKey("RGAPI-secret"), "euw1", None))  # type: ignore[arg-type]

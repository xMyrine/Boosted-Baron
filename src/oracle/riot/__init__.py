"""Riot API access: the only package that does HTTP or sees the API key. `_limiter` is private."""

from oracle.riot._limiter import Priority
from oracle.riot.endpoints import Endpoint, Host
from oracle.riot.errors import (
    AuthError,
    BadRequestError,
    DeadlineExceeded,
    ForbiddenError,
    KeyInUse,
    RiotError,
    TransientError,
)
from oracle.riot.gateway import GONE, Doc, Gone, RiotApi, RiotGateway, classify

__all__ = [
    "GONE", "AuthError", "BadRequestError", "DeadlineExceeded", "Doc", "Endpoint", "ForbiddenError", "Gone",
    "Host", "KeyInUse", "Priority", "RiotApi", "RiotError", "RiotGateway", "TransientError", "classify",
]

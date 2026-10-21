"""The closed set of errors the `riot` package raises. Callers never see HTTP status codes or 429s."""

from __future__ import annotations


class RiotError(Exception):
    pass


class TransientError(RiotError):
    """5xx, network failure, or too many 429s for one request. Retry later (the crawler records a backoff)."""


class DeadlineExceeded(RiotError):
    """An interactive call could not get a send permit before its deadline. Nothing was sent."""

    def __init__(self, eta_s: float | None) -> None:
        super().__init__(f"rate budget busy (eta {eta_s:.0f}s)" if eta_s is not None else "rate budget busy")
        self.eta_s = eta_s


class AuthError(RiotError):
    """401: the key is invalid or revoked. The gateway refuses every later call."""


class ForbiddenError(RiotError):
    """403 on one endpoint (e.g. not enabled for this key). That endpoint is disabled for this process."""


class BadRequestError(RiotError):
    """Any other 4xx: a bug on our side. Never retried."""


class KeyInUse(RiotError):
    """Another process on this machine holds the API-key lock (data/riot.lock)."""

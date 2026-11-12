"""Errors the bot raises for the operator. The CLI prints their message and exits 1; none carries a secret."""


class BotError(Exception):
    pass


class ModelError(BotError):
    """The model on disk cannot serve the running feature code. The message says what to retrain."""


class AnnounceError(BotError):
    """The message could not be delivered. The message says what to fix."""

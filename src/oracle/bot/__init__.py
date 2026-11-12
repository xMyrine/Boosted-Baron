"""The Discord bot: turns one live game into a stored, scored, rendered prophecy.

Import-light on purpose: `oracle crawl` and `oracle dataset` never load this package, and only `predictor`
imports scikit-learn (lazily) while only `announce` and `commands` import discord.py (lazily).
"""

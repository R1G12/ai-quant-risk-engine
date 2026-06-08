"""Display formatting for tracker monetary amounts."""

from __future__ import annotations

_SYMBOLS = {"USD": "$", "SGD": "S$", "EUR": "€", "GBP": "£"}


def money_symbol(currency: str) -> str:
    return _SYMBOLS.get(currency.upper(), currency.upper() + " ")


def format_money(amount: float, currency: str, *, decimals: int = 0) -> str:
    sym = money_symbol(currency)
    if decimals == 0:
        return f"{sym}{amount:,.0f}"
    return f"{sym}{amount:,.{decimals}f}"

"""Build data/input/portfolio/dummy_portfolio.xlsx (committed template)."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "input" / "portfolio" / "dummy_portfolio.xlsx"

HEADERS = [
    "trade_id",
    "ticker",
    "trade_date",
    "side",
    "action",
    "quantity",
    "price",
    "fees",
    "lot_id",
    "rolled_from_lot_id",
    "notes",
]

ROWS = [
    ["T001", "NVDA", "2025-06-02", "long", "buy", 10, 120.5, 1.0, "NVDA_L001", "", "Open long NVDA"],
    ["T002", "NVDA", "2025-08-15", "long", "sell", 5, 150.0, 1.0, "NVDA_L001", "", "Partial take profit"],
    ["T003", "SMH", "2025-05-10", "long", "buy", 20, 200.0, 2.0, "SMH_L001", "", "Open SMH"],
    ["T004", "SMH", "2025-09-01", "long", "sell", 20, 210.0, 2.0, "SMH_L001", "", "Full close SMH"],
    ["T005", "NVDA", "2025-09-20", "long", "sell", 5, 165.0, 1.0, "NVDA_L001", "", "Close remainder before roll"],
    ["T006", "NVDA", "2025-09-21", "long", "buy", 10, 166.0, 1.0, "NVDA_L002", "NVDA_L001", "Roll into new lot"],
    ["T007", "MSFT", "2025-07-01", "short", "sell", 15, 400.0, 1.5, "MSFT_S001", "", "Open short MSFT"],
    ["T008", "MSFT", "2025-10-10", "short", "buy", 15, 380.0, 1.5, "MSFT_S001", "", "Cover short MSFT"],
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "trades"
    ws.append(HEADERS)
    for row in ROWS:
        ws.append(row)
    wb.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

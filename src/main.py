"""Convenience entrypoint: python -m src.main (delegates to Typer CLI)."""

from src.cli import app

if __name__ == "__main__":
    app()

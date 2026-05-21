# run_test.py – end‑to‑end validation of the pipeline

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.ingestion.ingest import fetch_news
from src.preprocessing.preprocess import preprocess_news
from src.sentiment.finbert import run_sentiment

def main() -> None:
    print("[INFO] Starting pipeline validation…")
    raw_path = fetch_news()
    print(f"[INFO] Raw news generated at {raw_path}")
    processed_path = preprocess_news()
    print(f"[INFO] Preprocessed data written to {processed_path}")
    sentiment_path = run_sentiment()
    print(f"[INFO] Sentiment results saved to {sentiment_path}")

if __name__ == "__main__":
    main()

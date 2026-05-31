"""DVC stage: generate_returns."""

from __future__ import annotations

from src.features.returns import add_returns
from src.market.partitions import scan_market_dir
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.paths import FEATURES_RETURNS_DIR, METRICS_DIR, PROCESSED_MARKET_DIR, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(FEATURES_RETURNS_DIR)
    lf = scan_market_dir(PROCESSED_MARKET_DIR).pipe(add_returns)
    out = FEATURES_RETURNS_DIR / "returns.parquet"
    sink_lazy_parquet(lf, out, compression=app.market.compression)
    LOGGER.info("Returns features written", extra={"path": str(out)})
    log_stage_metrics(METRICS_DIR / "features_returns", {}, lazy_frame=lf)


if __name__ == "__main__":
    run()

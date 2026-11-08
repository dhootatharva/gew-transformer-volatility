"""Download daily index closes from Yahoo Finance and cache them locally."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gewvol.config import DataConfig

log = logging.getLogger(__name__)


def _download_close(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance >= 0.2.48 returns MultiIndex columns
        close = close.iloc[:, 0]
    return close.dropna()


def load_prices(cfg: DataConfig, refresh: bool = False) -> dict[str, pd.Series]:
    """Return {market: close series}, reading from ``cfg.cache_dir/raw`` when cached."""
    raw_dir = Path(cfg.cache_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prices: dict[str, pd.Series] = {}
    for name, ticker in cfg.tickers.items():
        path = raw_dir / f"{name}_{cfg.start}_{cfg.end}.csv"
        if path.exists() and not refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            log.info("Downloading %s (%s)", name, ticker)
            s = _download_close(ticker, cfg.start, cfg.end)
            s.rename(name).to_frame().to_csv(path)
        s.name = name
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        prices[name] = s
    return prices


def align_markets(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Inner-join markets on trading date, so only days on which all markets traded remain."""
    return pd.concat(list(prices.values()), axis=1, join="inner").dropna()

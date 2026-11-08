"""All experiment settings in one place.

The defaults reproduce the 60-day configuration of the thesis notebook:
60-day realized-volatility window, 60-step look-back, three markets,
70/15/15 chronological split, seed 42.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    tickers: dict[str, str] = field(
        default_factory=lambda: {"NIFTY": "^NSEI", "SP500": "^GSPC", "KOSPI": "^KS11"}
    )
    target_market: str = "NIFTY"
    start: str = "1995-01-01"
    end: str = "2025-01-01"
    rv_window: int = 60          # h: rolling window of the realized-volatility target
    lookback: int = 60           # L: input sequence length
    sigma_burn_in: int = 50      # first t for which GARCH/EGARCH/EWMA are fitted
    econ_clip: float = 0.1       # returns are clipped to +-10% before econometric fits
    feature_clip: float = 5.0    # clip applied when cleaning feature tensors
    train_frac: float = 0.70
    val_frac: float = 0.15
    cache_dir: Path = REPO_ROOT / "data"


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    head_size: int = 32
    epochs: int = 20
    lr: float = 1e-3             # full-batch Adam, MSE loss


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 160
    n_heads: int = 4
    num_layers: int = 4
    dim_ff: int = 320
    dropout: float = 0.05
    head_size: int = 64
    max_len: int = 500
    epochs: int = 60
    lr: float = 5e-5
    batch_size: int = 64         # Adam, Huber loss, grad-norm clip 1.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class Config:
    seed: int = 42
    # "train": normalisation statistics are fitted on the training split only (default).
    # "global": statistics over the full sample, as in the original notebook (leaks test data).
    normalisation: str = "train"
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    results_dir: Path = REPO_ROOT / "results"

    def to_dict(self) -> dict:
        d = asdict(self)
        return _stringify_paths(d)


def _stringify_paths(obj):
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        try:
            return obj.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return obj.name
    return obj

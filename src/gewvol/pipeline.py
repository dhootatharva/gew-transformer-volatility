"""End-to-end experiment: data -> GEW features -> baselines, LSTM and Transformer -> tables and figures."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gewvol.config import Config
from gewvol.data.features import compute_features
from gewvol.data.loader import align_markets, load_prices
from gewvol.data.sequences import (
    build_gew_sequences,
    build_pure_sequences,
    chrono_split,
    clean_and_normalize,
)
from gewvol.data.sigma import sigma_for_markets
from gewvol.evaluation.attention import lag_importance, market_spillover
from gewvol.evaluation.metrics import evaluate_all
from gewvol.evaluation.stat_tests import dm_test
from gewvol.training.train_lstm import run_lstm_experiment
from gewvol.training.train_transformer import run_transformer_experiment
from gewvol.visualization import plots

log = logging.getLogger(__name__)

ECON_NAMES = ("GARCH(1,1)", "EGARCH(1,1)", "EWMA")
DM_HAC_LAGS = 10  # Newey-West lags for the autocorrelation-robust DM variant
DM_PAIRS = [
    ("Transformer (GEW)", "LSTM (GEW)"),
    ("Transformer (GEW)", "Transformer (pure)"),
    ("LSTM (GEW)", "LSTM (pure)"),
    ("Transformer (GEW)", "LSTM (pure)"),
    ("Transformer (GEW)", "GARCH(1,1)"),
]


def prepare_data(cfg: Config, refresh: bool = False) -> dict:
    d = cfg.data
    prices = align_markets(load_prices(d, refresh=refresh))
    returns, rv = compute_features(prices, d.rv_window)
    sigma = sigma_for_markets(returns, d.sigma_burn_in, d.econ_clip, d.cache_dir)
    gew = build_gew_sequences(returns, rv, sigma, d.target_market, d.lookback, d.sigma_burn_in)
    pure = build_pure_sequences(returns, rv, d.target_market, d.lookback, d.sigma_burn_in)
    splits = chrono_split(len(gew.y), d.train_frac, d.val_frac)
    fit_end = None if cfg.normalisation == "global" else splits[0].stop
    gew.X = clean_and_normalize(gew.X, fit_end, d.feature_clip, gew_variant=True)
    pure.X = clean_and_normalize(pure.X, fit_end, d.feature_clip, gew_variant=False)
    return {"prices": prices, "returns": returns, "rv": rv, "sigma": sigma,
            "gew": gew, "pure": pure, "splits": splits}


def econometric_baselines(data: dict, cfg: Config) -> dict[str, np.ndarray]:
    """Annualised one-step sigma forecasts for the target market on the test samples.

    For a sample with origin t the forecast uses returns up to day t-1, the same
    information set as the neural models' input window.
    """
    d = cfg.data
    test = data["splits"][2]
    t_idx = data["gew"].t_index[test]
    g, e, w = data["sigma"][d.target_market]
    j = t_idx - d.sigma_burn_in
    return {name: arr[j] * np.sqrt(252) for name, arr in zip(ECON_NAMES, (g, e, w))}


def _model_specs(pure, gew):
    return [
        ("LSTM (pure)", pure, "lstm"), ("LSTM (GEW)", gew, "lstm"),
        ("Transformer (pure)", pure, "tf"), ("Transformer (GEW)", gew, "tf"),
    ]


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _save_run(path: Path, res: dict) -> None:
    """Cache a trained model's test outputs so the four models can be trained in parallel processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in res.items() if k != "history"}
    np.savez_compressed(path, history=json.dumps(res["history"]), **arrays)


def _load_run(path: Path) -> dict:
    z = np.load(path)
    res = {k: z[k] for k in z.files if k != "history"}
    res["history"] = json.loads(str(z["history"]))
    return res


def run(cfg: Config, out_dir: Path | None = None, refresh: bool = False,
        only: list[str] | None = None) -> dict:
    """Run the experiment. ``only`` trains (and caches) just the named models and skips reporting."""
    t0 = time.time()
    out_dir = Path(out_dir or cfg.results_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(cfg, refresh=refresh)
    t_data = time.time() - t0
    gew, pure, splits = data["gew"], data["pure"], data["splits"]
    n_markets = len(cfg.data.tickers)
    target_stats_on_train = cfg.normalisation != "global"
    y_test = gew.y[splits[2]]
    dates_test = gew.dates[splits[2]]
    assert np.allclose(y_test, pure.y[splits[2]]), "GEW and pure test targets differ"

    preds: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {"data_and_gew_features_s": t_data}
    for name, arr in econometric_baselines(data, cfg).items():
        preds[name] = arr

    runs = {}
    for label, ds, kind in _model_specs(pure, gew):
        if only and label not in only:
            continue
        cache = out_dir / "runs" / f"{_slug(label)}.npz"
        if cache.exists():
            log.info("Loading cached %s from %s", label, cache.name)
            res = _load_run(cache)
        else:
            s = time.time()
            log.info("Training %s on %s", label, ds.X.shape)
            if kind == "lstm":
                res = run_lstm_experiment(ds.X, ds.y, splits, cfg.lstm, cfg.seed)
            else:
                res = run_transformer_experiment(ds.X, ds.y, splits, cfg.transformer, cfg.seed,
                                                 n_markets, target_stats_on_train)
            res["train_seconds"] = time.time() - s
            _save_run(cache, res)
        timings[f"{label}_s"] = float(res["train_seconds"])
        runs[label] = res
        preds[label] = res["pred"]
    if only:
        return {"runs": runs}

    # Metrics. EGARCH fits can fail (NaN); evaluate each model on its finite forecasts.
    rows, n_eval = {}, {}
    for name, p in preds.items():
        mask = np.isfinite(p)
        rows[name] = evaluate_all(y_test[mask], p[mask])
        n_eval[name] = int(mask.sum())
    metrics = pd.DataFrame(rows).T
    metrics["n_test"] = pd.Series(n_eval)
    metrics.index.name = "model"

    dm_rows = []
    for a, b in DM_PAIRS:
        mask = np.isfinite(preds[a]) & np.isfinite(preds[b])
        stat, p = dm_test(y_test[mask], preds[a][mask], preds[b][mask])
        stat_nw, p_nw = dm_test(y_test[mask], preds[a][mask], preds[b][mask], hac_lags=DM_HAC_LAGS)
        dm_rows.append({"model_1": a, "model_2": b, "dm_stat": stat, "p_value": p,
                        "dm_stat_nw": stat_nw, "p_value_nw": p_nw})
    dm = pd.DataFrame(dm_rows)

    markets = list(cfg.data.tickers)
    spill_gew = market_spillover(runs["Transformer (GEW)"]["attn"], n_markets)
    spill_pure = market_spillover(runs["Transformer (pure)"]["attn"], n_markets)
    spill_df = pd.concat({
        "Transformer (GEW)": pd.DataFrame(spill_gew, index=markets, columns=markets),
        "Transformer (pure)": pd.DataFrame(spill_pure, index=markets, columns=markets),
    })

    _write_outputs(out_dir, fig_dir, cfg, metrics, dm, spill_df, preds, y_test, dates_test, runs,
                   markets, n_markets)
    timings["total_wall_s"] = time.time() - t0
    timings["sum_of_model_training_s"] = sum(v for k, v in timings.items() if k.endswith("_s") and "(" in k)
    info = {
        "config": cfg.to_dict(),
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        "data": {
            "aligned_days": int(len(data["prices"])),
            "first_day": str(data["prices"].index[0].date()),
            "last_day": str(data["prices"].index[-1].date()),
            "samples": int(len(gew.y)),
            "train/val/test": [int(s.stop - s.start) for s in splits],
            "test_period": [str(dates_test[0].date()), str(dates_test[-1].date())],
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    return {"metrics": metrics, "dm": dm, "spillover": spill_df, "info": info}


def _write_outputs(out_dir, fig_dir, cfg, metrics, dm, spill_df, preds, y_test, dates_test, runs,
                   markets, n_markets) -> None:
    metrics.to_csv(out_dir / "metrics.csv", float_format="%.6f")
    dm.to_csv(out_dir / "dm_tests.csv", index=False, float_format="%.6g")
    spill_df.to_csv(out_dir / "market_spillover.csv", float_format="%.4f")
    pd.DataFrame({"date": dates_test, "actual": y_test, **preds}).to_csv(
        out_dir / "test_predictions.csv", index=False, float_format="%.6f")
    pd.DataFrame({k: pd.DataFrame(v["history"]).set_index("epoch").stack() for k, v in runs.items()}
                 ).to_csv(out_dir / "training_history.csv", float_format="%.6f")

    cols = ["RMSE", "MAE", "HMAE", "HMSE", "QLIKE"]
    md = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    best = metrics[cols].min()
    for name, row in metrics[cols].iterrows():
        cells = [f"**{row[c]:.6f}**" if np.isclose(row[c], best[c]) else f"{row[c]:.6f}" for c in cols]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", f"| Comparison | DM statistic | p-value | DM (Newey-West, {DM_HAC_LAGS} lags) | p-value |",
           "|---|---:|---:|---:|---:|"]
    for _, r in dm.iterrows():
        md.append(f"| {r.model_1} vs {r.model_2} | {r.dm_stat:.3f} | {r.p_value:.2e} | "
                  f"{r.dm_stat_nw:.3f} | {r.p_value_nw:.2e} |")
    (out_dir / "results.md").write_text("\n".join(md) + "\n")

    deep = {k: preds[k] for k in ("Transformer (GEW)", "LSTM (GEW)", "Transformer (pure)", "LSTM (pure)")}
    plots.plot_predictions(dates_test, y_test, deep, fig_dir / "predictions_vs_actual.png",
                           f"{cfg.data.target_market} {cfg.data.rv_window}-day realized volatility, test period")
    plots.plot_metric_bars(metrics, ["RMSE", "QLIKE"], fig_dir / "metrics.png")
    attn = runs["Transformer (GEW)"]["attn"]
    plots.plot_market_spillover(spill_df.loc["Transformer (GEW)"].to_numpy(), markets,
                                fig_dir / "market_spillover.png", "GEW-Transformer cross-market attention")
    plots.plot_attention_heatmap(attn, n_markets, fig_dir / "attention_heatmap.png",
                                 "GEW-Transformer attention (test mean)")
    plots.plot_lag_importance({
        "Transformer (GEW)": lag_importance(attn, n_markets),
        "Transformer (pure)": lag_importance(runs["Transformer (pure)"]["attn"], n_markets),
    }, fig_dir / "lag_importance.png")


def smoke_config(cfg: Config) -> Config:
    """One epoch per model; only for checking that the pipeline runs end to end."""
    return replace(cfg, lstm=replace(cfg.lstm, epochs=1), transformer=replace(cfg.transformer, epochs=1))

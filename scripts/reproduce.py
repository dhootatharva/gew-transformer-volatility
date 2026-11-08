"""Reproduce the headline results table, Diebold-Mariano tests and figures.

    python scripts/reproduce.py                        # leakage-free run -> results/
    python scripts/reproduce.py --normalisation global # original notebook normalisation -> results/notebook_normalisation/
    python scripts/reproduce.py --smoke                # 1 epoch per model, end-to-end check -> results/smoke/
    python scripts/reproduce.py --only "Transformer (GEW)" --threads 2   # train one model, cache it
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

import pandas as pd
import torch

from gewvol.config import Config
from gewvol.pipeline import run, smoke_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normalisation", choices=["train", "global"], default="train",
                    help="fit feature/target normalisation on the training split (default) "
                         "or on the full sample as in the original notebook")
    ap.add_argument("--smoke", action="store_true", help="1 epoch per model (pipeline check only)")
    ap.add_argument("--refresh-data", action="store_true", help="re-download prices from Yahoo Finance")
    ap.add_argument("--out", default=None, help="output directory (default depends on the mode)")
    ap.add_argument("--only", action="append", default=None, metavar="MODEL",
                    help='train and cache only this model, e.g. --only "Transformer (GEW)"; '
                         "repeatable. Lets the models train in parallel processes; a later run "
                         "without --only loads the cached models and writes the tables")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = replace(Config(), normalisation=args.normalisation)
    out = cfg.results_dir
    if args.normalisation == "global":
        out = out / "notebook_normalisation"
    if args.smoke:
        cfg, out = smoke_config(cfg), cfg.results_dir / "smoke"
    result = run(cfg, out_dir=args.out or out, refresh=args.refresh_data, only=args.only)
    if args.only:
        print(f"Cached {args.only} in {args.out or out}/runs")
        return

    with pd.option_context("display.float_format", "{:.6f}".format, "display.width", 140):
        print(result["metrics"])
        print(result["dm"])
        print(result["spillover"])
    print(f"Outputs written to {args.out or out}")


if __name__ == "__main__":
    main()

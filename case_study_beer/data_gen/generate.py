"""Write the beer demo data to disk.

    python -m data_gen.generate --out data/                     # default case study data
    python -m data_gen.generate --out data/ --outcome reorder   # binary outcome
    python -m data_gen.generate --out data/ --generic            # also the generic panel

Files written:
    beer_panel.csv     one row per outlet-month: features, arm, rule_arm, outcomes
    beer_truth.csv     true effects, expected outcomes, propensities, net values, best arm
    beer_meta.json     arm labels, costs, capacities, value per outcome, months, counts
    generic_panel.csv / generic_truth.csv   with --generic
Run from the case_study_beer folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .beer_promo import make_beer_promo_data
from .synthetic import make_multi_arm_data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    ap.add_argument("--n-outlets", type=int, default=20_000)
    ap.add_argument("--n-months", type=int, default=6)
    ap.add_argument("--randomised-share", type=float, default=0.7)
    ap.add_argument("--outcome", choices=("volume", "reorder"), default="volume")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--generic", action="store_true", help="also write the generic panel")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, truth, meta = make_beer_promo_data(n_outlets=args.n_outlets, n_months=args.n_months,
                                           randomised_share=args.randomised_share, outcome=args.outcome,
                                           random_state=args.random_state)
    df.to_csv(out / "beer_panel.csv", index=False)
    truth.to_csv(out / "beer_truth.csv", index=False)
    (out / "beer_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"beer panel: {len(df):,} rows x {df.shape[1]} columns -> {out}")
    if args.generic:
        g, gt = make_multi_arm_data(random_state=args.random_state)
        g.to_csv(out / "generic_panel.csv", index=False)
        gt.to_csv(out / "generic_truth.csv", index=False)
        print(f"generic panel: {len(g):,} rows -> {out}")


if __name__ == "__main__":
    main()

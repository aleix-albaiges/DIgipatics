#!/usr/bin/env python3
"""
Select best checkpoint(s) from a fold-based training directory.

Designed for directories like:
  checkpoints_conch_masklut/
    metrics.csv
    best_Val1_0.6774.pth
    best_Val2_0.6931.pth
    ...

Outputs:
  - best_per_fold.csv
  - checkpoint_ranking.csv
  - checkpoint_selection.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import sicap_imports  # noqa: F401
from paths import default_checkpoint_dir


CKPT_REGEX = re.compile(r"^best_(?P<fold>[^_]+)_(?P<score>\d+\.\d+)\.pth$")


def _read_metrics(metrics_path: Path) -> Optional[pd.DataFrame]:
    if not metrics_path.exists():
        return None

    df = pd.read_csv(metrics_path)
    required_cols = {"fold", "epoch", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        return None

    return df


def _scan_checkpoints(checkpoints_dir: Path) -> Dict[str, List[Tuple[float, Path]]]:
    by_fold: Dict[str, List[Tuple[float, Path]]] = {}
    for path in checkpoints_dir.rglob("*.pth"):
        m = CKPT_REGEX.match(path.name)
        if not m:
            continue
        fold = m.group("fold")
        score = float(m.group("score"))
        by_fold.setdefault(fold, []).append((score, path))

    for fold in by_fold:
        by_fold[fold].sort(key=lambda x: x[0], reverse=True)

    return by_fold


def _best_from_filenames(ckpts_by_fold: Dict[str, List[Tuple[float, Path]]]) -> pd.DataFrame:
    rows = []
    for fold, candidates in ckpts_by_fold.items():
        if not candidates:
            continue
        score, path = candidates[0]
        rows.append(
            {
                "fold": fold,
                "epoch": None,
                "macro_f1": float(score),
                "checkpoint_path": str(path),
                "source": "filename",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["fold", "epoch", "macro_f1", "checkpoint_path", "source"])
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def _pick_ckpt_for_row(
    fold: str,
    target_macro_f1: float,
    ckpts_by_fold: Dict[str, List[Tuple[float, Path]]],
    tolerance: float = 1e-4,
) -> Optional[Path]:
    candidates = ckpts_by_fold.get(fold, [])
    if not candidates:
        return None

    # Prefer exact-ish match to macro_f1 in CSV (filenames store 4 decimals).
    best = min(candidates, key=lambda x: abs(x[0] - target_macro_f1))
    if abs(best[0] - target_macro_f1) <= (tolerance + 5e-4):
        return best[1]

    # Fallback: highest checkpoint score in this fold.
    return candidates[0][1]


def _best_rows_per_fold(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("fold")["macro_f1"].idxmax()
    out = df.loc[idx].copy()
    out = out.sort_values("fold").reset_index(drop=True)
    return out


def _recommendations(best_df: pd.DataFrame) -> Dict[str, object]:
    def _safe_epoch(v):
        if pd.isna(v):
            return None
        return int(v)

    if best_df.empty:
        return {}

    # Highest validation macro_f1 among fold winners.
    single_best_idx = best_df["macro_f1"].idxmax()
    single_best = best_df.loc[single_best_idx]

    # "Robust" choice: best fold nearest to median fold performance.
    median_val = float(best_df["macro_f1"].median())
    robust_idx = (best_df["macro_f1"] - median_val).abs().idxmin()
    robust_best = best_df.loc[robust_idx]

    return {
        "single_best": {
            "fold": str(single_best["fold"]),
            "epoch": _safe_epoch(single_best["epoch"]),
            "macro_f1": float(single_best["macro_f1"]),
            "checkpoint": str(single_best.get("checkpoint_path", "")),
        },
        "robust_best": {
            "fold": str(robust_best["fold"]),
            "epoch": _safe_epoch(robust_best["epoch"]),
            "macro_f1": float(robust_best["macro_f1"]),
            "checkpoint": str(robust_best.get("checkpoint_path", "")),
        },
        "fold_stats": {
            "n_folds": int(best_df["fold"].nunique()),
            "mean_macro_f1": float(best_df["macro_f1"].mean()),
            "std_macro_f1": float(best_df["macro_f1"].std(ddof=0)),
            "median_macro_f1": median_val,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank fold checkpoints and recommend best models.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=default_checkpoint_dir("checkpoints_conch_masklut"),
        help="Directory containing metrics.csv and best_*.pth checkpoints.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics.csv",
        help="Metrics CSV filename inside checkpoints directory.",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoints_dir
    ckpts_by_fold = _scan_checkpoints(ckpt_dir)
    if not ckpts_by_fold:
        raise FileNotFoundError(f"No best_*.pth found under: {ckpt_dir}")

    # Primary source: all checkpoint filenames found in directory.
    best_df = _best_from_filenames(ckpts_by_fold)

    # Optional enrichment from metrics.csv (if present and valid).
    metrics_path = ckpt_dir / args.metrics_file
    df = _read_metrics(metrics_path)
    if df is not None:
        metrics_best = _best_rows_per_fold(df)
        if not metrics_best.empty:
            metrics_best["checkpoint_path"] = metrics_best.apply(
                lambda r: _pick_ckpt_for_row(
                    fold=str(r["fold"]),
                    target_macro_f1=float(r["macro_f1"]),
                    ckpts_by_fold=ckpts_by_fold,
                ),
                axis=1,
            )
            metrics_best["checkpoint_path"] = metrics_best["checkpoint_path"].apply(
                lambda p: str(p) if p is not None else ""
            )
            metrics_best["source"] = "metrics"

            # Keep the highest macro_f1 per fold regardless of source.
            # If there is a tie, prefer metrics row because it includes epoch info.
            merged = pd.concat([best_df, metrics_best], ignore_index=True)
            merged["src_priority"] = merged["source"].map({"metrics": 0, "filename": 1}).fillna(1)
            merged = merged.sort_values(["fold", "macro_f1", "src_priority"], ascending=[True, False, True])
            best_df = merged.groupby("fold", as_index=False).nth(0).reset_index(drop=True)
            best_df = best_df.drop(columns=["src_priority"])

    ranking_df = best_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    best_per_fold_csv = ckpt_dir / "best_per_fold.csv"
    ranking_csv = ckpt_dir / "checkpoint_ranking.csv"
    summary_json = ckpt_dir / "checkpoint_selection.json"

    best_df.to_csv(best_per_fold_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)

    summary = _recommendations(ranking_df)
    summary["files"] = {
        "best_per_fold_csv": str(best_per_fold_csv),
        "checkpoint_ranking_csv": str(ranking_csv),
        "selection_json": str(summary_json),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote: {best_per_fold_csv}")
    print(f"[OK] Wrote: {ranking_csv}")
    print(f"[OK] Wrote: {summary_json}")

    if summary:
        sb = summary["single_best"]
        rb = summary["robust_best"]
        print("\nRecommended checkpoints:")
        print(
            f"- single_best : fold={sb['fold']} epoch={sb['epoch']} macro_f1={sb['macro_f1']:.4f} "
            f"path={sb['checkpoint'] or '<not found>'}"
        )
        print(
            f"- robust_best : fold={rb['fold']} epoch={rb['epoch']} macro_f1={rb['macro_f1']:.4f} "
            f"path={rb['checkpoint'] or '<not found>'}"
        )
        stats = summary["fold_stats"]
        print(
            f"- fold_stats  : n={stats['n_folds']} mean={stats['mean_macro_f1']:.4f} "
            f"std={stats['std_macro_f1']:.4f} median={stats['median_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

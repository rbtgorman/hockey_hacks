"""
Durable results export for xG model runs.

WHY THIS EXISTS
---------------
Every train_*.py script currently prints its metrics to stdout and nothing
else. That means the only record of "v2.3 hit AUC 0.7706 with a 0.0178 max
calibration gap" lives in a terminal scrollback buffer. Nothing is committed,
nothing is comparable across runs, and nothing is visible to anyone who did
not run the script themselves.

This module writes each run to results/<version>/ as:

    metrics.json            machine-readable run record (metrics, params, commit)
    reliability.csv         the decile reliability table
    calibration.png         reliability diagram, for embedding in the README
    feature_importance.csv  LightGBM gain importance

and regenerates results/leaderboard.md from every metrics.json on disk, so the
version comparison table is generated rather than hand-maintained.

A SECOND, LESS OBVIOUS REASON
-----------------------------
train_v1.py, train_v2_1.py, train_v2_2.py and train_v2_3.py each define their
own local copy of reliability_table(). Max calibration gap is only comparable
across versions if the binning is byte-identical in all of them. Four copies
drifting apart is a silent way to make "v2.3 has a better max gap than v1"
untrue. reliability_table() lives here now. Import it; do not re-implement it.

USAGE
-----
    from model.reporting import reliability_table, write_results, build_leaderboard

    rel = reliability_table(y_test, p_test)
    write_results(version="v2.3", ...)

    python -m model.reporting --leaderboard    # rebuild results/leaderboard.md

CALIBRATION STATS AND POPULATIONS
---------------------------------
Max calibration gap cannot tell a level shift from a shape problem, and on
2024-25 the shots-on-goal models suffered a level shift (the NHL started
recording goalie-touched wide pucks as missed shots). calibration_stats()
reports observed/expected goals (O/E) and the calibration slope separately.
Training scripts put them inside each split's metrics dict:

    splits["test"].update(calibration_stats(y_test, p_test))

Runs also declare their shot population in data_summary["population"]
("shots_on_goal" if absent, "fenwick" for unblocked attempts). The
leaderboard renders one table per population, because AUC and log loss are
not comparable across different sets of shots.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: this runs over SSH with no display
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Ice blue / slate, matching the planned frontend palette.
ACCENT = "#2a7fbf"
SLATE = "#334155"
GRID = "#d6dde5"


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def reliability_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Decile reliability table: equal-count bins on predicted probability.

    qcut (equal *count*) rather than cut (equal *width*) is deliberate. Goal
    probabilities are heavily right-skewed -- most shots sit under 0.08 -- so
    equal-width bins would put ~95% of shots in bin 0 and leave the top bins
    with too few events to estimate a rate from. Equal-count bins keep every
    bin statistically meaningful.

    duplicates="drop" guards the case where a model is confident enough that
    bin edges collide; you get fewer than n_bins rows rather than an exception.
    """
    df = pd.DataFrame({"y": list(y_true), "p": list(y_pred)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    grp = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index()
    )
    grp["pred_vs_actual_gap"] = grp["mean_pred"] - grp["actual_rate"]
    return grp


def max_abs_gap(rel: pd.DataFrame) -> float:
    return float(rel["pred_vs_actual_gap"].abs().max())


def calibration_stats(y_true, y_pred) -> dict:
    """Calibration-in-the-large and calibration slope for one split.

      o_e            observed / expected goals. 1.00 = right on average;
                     1.07 = 7% more goals than predicted.
      citl           mean_pred - observed_rate. Same sign as the reliability
                     table's gap column: negative = under-predicting.
      cal_slope      slope of a logistic fit of the outcome on logit(p).
                     1.0 = spread is right; below 1 = predictions too
                     extreme (the usual overfitting signature); above 1 =
                     too timid.
      cal_intercept  that fit's intercept.

    The fit is a two-parameter Newton-Raphson in numpy, so it does not
    depend on the installed scikit-learn's penalty API. It starts at
    (0, 1), perfect calibration, which is close to any usable model.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1.0 - p))
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = mu * (1.0 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    mean_pred = float(p.mean())
    observed = float(y.mean())
    return {
        "mean_pred": mean_pred,
        "observed_rate": observed,
        "o_e": observed / mean_pred,
        "citl": mean_pred - observed,
        "cal_intercept": float(beta[0]),
        "cal_slope": float(beta[1]),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def calibration_plot(rel: pd.DataFrame, version: str, out_path: Path,
                     auc: float | None = None) -> None:
    """Reliability diagram: mean predicted probability vs observed rate per decile."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=150)

    lo = float(min(rel["mean_pred"].min(), rel["actual_rate"].min()))
    hi = float(max(rel["mean_pred"].max(), rel["actual_rate"].max()))
    pad = (hi - lo) * 0.08 or 0.01
    lims = (max(0.0, lo - pad), hi + pad)

    ax.plot(lims, lims, linestyle="--", linewidth=1.2, color=SLATE,
            alpha=0.6, label="perfect calibration", zorder=1)

    # Fixed marker size on purpose: qcut gives equal-count bins, so scaling
    # marker area by n would encode nothing while implying it encodes something.
    n_per_bin = int(rel["n"].median())
    ax.plot(rel["mean_pred"], rel["actual_rate"], color=ACCENT,
            linewidth=1.4, alpha=0.85, zorder=2)
    ax.scatter(rel["mean_pred"], rel["actual_rate"], s=90, color=ACCENT,
               edgecolor="white", linewidth=0.9, zorder=3,
               label=f"decile (~{n_per_bin:,} shots each)")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Mean predicted goal probability")
    ax.set_ylabel("Observed goal rate")

    gap = max_abs_gap(rel)
    subtitle = f"held-out season  |  max |pred - actual| = {gap:.4f}"
    if auc is not None:
        subtitle = f"held-out season  |  AUC {auc:.4f}  |  max gap {gap:.4f}"
    ax.set_title(f"xG {version} calibration\n{subtitle}", fontsize=11, color=SLATE)

    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

def _git_commit() -> str | None:
    """Short commit hash, or None if git is unavailable or the tree isn't a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    """True if there are uncommitted changes. A metric produced from a dirty
    tree is not reproducible from the recorded commit; record that fact."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None  # not a git repo / git unavailable -> unknown, not "clean"
        return bool(out.stdout.strip())
    except Exception:
        return None


REQUIRED_SPLIT_KEYS = {"auc", "pr_auc", "log_loss", "brier"}


def _validate_splits(splits) -> None:
    """Fail loudly and legibly on a malformed splits argument.

    Without this, passing the wrong shape produces errors like
    "float() argument must be a string or a real number, not 'NoneType'"
    several frames deep, which tells you nothing about what you got wrong.
    The common mistakes are naming the metric "roc_auc" instead of "auc",
    and putting scalars at the top level alongside the split dicts.
    """
    if not isinstance(splits, Mapping):
        raise TypeError(
            f"splits must be a mapping of split name -> metrics dict, got "
            f"{type(splits).__name__}. Expected shape: "
            '{"train": {"auc": ..., "pr_auc": ..., "log_loss": ..., "brier": ...}, "val": {...}, "test": {...}}'
        )
    for name, m in splits.items():
        if not isinstance(m, Mapping):
            raise TypeError(
                f'splits["{name}"] must be a metrics dict, got {type(m).__name__}. '
                "Scalars belong inside a split, not alongside them - the max calibration "
                "gap is computed from the reliability table and must not be passed here."
            )
        missing = REQUIRED_SPLIT_KEYS - set(m)
        if missing:
            raise KeyError(
                f'splits["{name}"] is missing {sorted(missing)}. '
                f"Got {sorted(m)}. Note the key is \"auc\", not \"roc_auc\" - "
                "the leaderboard reads splits.test.auc and renders an em-dash if absent."
            )


def write_results(
    version: str,
    description: str,
    splits: Mapping[str, Mapping[str, float]],
    reliability: pd.DataFrame,
    feature_importance: pd.DataFrame | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    features: Sequence[str] | None = None,
    data_summary: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    """Write one run to results/<version>/ and refresh the leaderboard.

    splits: {"train": {"auc":..., "pr_auc":..., "log_loss":..., "brier":...,
                       "n":..., "goal_rate":...}, "val": {...}, "test": {...}}
    """
    _validate_splits(splits)

    out_dir = RESULTS_DIR / version.replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_auc = float(splits["test"]["auc"]) if "test" in splits else None
    gap = max_abs_gap(reliability)

    record = {
        "version": version,
        "description": description,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "data": dict(data_summary or {}),
        "splits": {k: dict(v) for k, v in splits.items()},
        "calibration": {
            "n_bins": int(len(reliability)),
            "max_abs_gap": gap,
            "binning": "equal-count (qcut) on predicted probability",
        },
        "hyperparameters": dict(hyperparameters or {}),
        "features": list(features or []),
        "notes": notes,
    }

    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    reliability.to_csv(out_dir / "reliability.csv", index=False)
    calibration_plot(reliability, version, out_dir / "calibration.png", auc=test_auc)

    if feature_importance is not None:
        feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\nResults written to {out_dir}/")
    print("  metrics.json  reliability.csv  calibration.png"
          + ("  feature_importance.csv" if feature_importance is not None else ""))

    build_leaderboard()
    return out_dir


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _version_sort_key(v: str):
    """Sort v1 < v2.1 < v2.3 < v2.10 numerically, not lexically.

    Only the digit groups count, so suffixed names like 'v1-fenwick' sort by
    their number as well.
    """
    return [int(x) for x in re.findall(r"\d+", v)] or [0]


POPULATIONS = {
    "shots_on_goal": (
        "Shots on goal (legacy population)",
        "Trained and scored on shots on goal, including regular-season shootout "
        "attempts (about 0.6% of rows). From 2023-24 the NHL records many "
        "goalie-touched wide pucks as missed shots, so this population's "
        "conversion drifts upward across the split. Kept for reproducibility.",
    ),
    "fenwick": (
        "Unblocked attempts (Fenwick)",
        "Shots on goal plus missed shots. Shootout attempts are excluded, as are "
        "events logged in only part of the window (teammate blocks, failed bank "
        "attempts). Goals per unblocked attempt hold steady across the split "
        "within each distance band. AUC and log loss are not comparable with "
        "the shots-on-goal table: the set of shots differs.",
    ),
}


def build_leaderboard() -> Path | None:
    """Regenerate results/leaderboard.md from every results/*/metrics.json.

    One table per shot population, each ordered chronologically by version,
    not by AUC. The iteration story is the point: what each change was trying
    to do and what it actually did.
    """
    if not RESULTS_DIR.exists():
        return None

    rows = []
    for mpath in sorted(RESULTS_DIR.glob("*/metrics.json")):
        try:
            rec = json.loads(mpath.read_text())
        except Exception as e:
            print(f"  [warn] skipping unreadable {mpath}: {e}")
            continue
        test = rec.get("splits", {}).get("test", {})
        rows.append({
            "version": rec.get("version", mpath.parent.name),
            "description": rec.get("description", ""),
            "population": (rec.get("data") or {}).get("population", "shots_on_goal"),
            "auc": test.get("auc"),
            "pr_auc": test.get("pr_auc"),
            "log_loss": test.get("log_loss"),
            "brier": test.get("brier"),
            "max_gap": rec.get("calibration", {}).get("max_abs_gap"),
            "o_e": test.get("o_e"),
            "cal_slope": test.get("cal_slope"),
            "commit": rec.get("git_commit"),
        })

    if not rows:
        return None

    rows.sort(key=lambda r: _version_sort_key(r["version"]))

    def fmt(x, nd=4):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"

    lines = [
        "# Model leaderboard",
        "",
        "Generated by `python -m model.reporting --leaderboard`. Do not edit by hand.",
        "",
        "All metrics are on the **held-out 2024-25 test season**. Training is",
        "2022-23, validation 2023-24. No random splits at any point.",
        "",
        "O/E is observed goals divided by predicted goals (1.00 is right on",
        "average). Calib. slope is the logistic slope of the outcome on logit(p)",
        "(1.00 is ideal; below 1 means over-confident). Both are blank for runs",
        "recorded before they existed.",
        "",
    ]

    present = {r["population"] for r in rows}
    order = [p for p in POPULATIONS if p in present] + sorted(present - set(POPULATIONS))
    for pop in order:
        title, note = POPULATIONS.get(pop, (pop, ""))
        lines += [f"## {title}", ""]
        if note:
            lines += [note, ""]
        lines += [
            "| Version | Test AUC | PR-AUC | Log loss | Brier | Max calib. gap "
            "| O/E | Calib. slope | Commit |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in (r for r in rows if r["population"] == pop):
            lines.append(
                f"| **{r['version']}** | {fmt(r['auc'])} | {fmt(r['pr_auc'])} | "
                f"{fmt(r['log_loss'])} | {fmt(r['brier'])} | {fmt(r['max_gap'])} | "
                f"{fmt(r['o_e'], 3)} | {fmt(r['cal_slope'], 3)} | "
                f"`{r['commit'] or '—'}` |"
            )
        lines.append("")

    lines += ["## What each version changed", ""]
    for r in rows:
        if r["description"]:
            lines.append(f"- **{r['version']}** — {r['description']}")
    lines.append("")

    out = RESULTS_DIR / "leaderboard.md"
    out.write_text("\n".join(lines))
    print(f"Leaderboard written to {out}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="xG results reporting utilities")
    ap.add_argument("--leaderboard", action="store_true",
                    help="Rebuild results/leaderboard.md from existing metrics.json files")
    args = ap.parse_args()

    if args.leaderboard:
        if build_leaderboard() is None:
            print(f"No metrics.json files found under {RESULTS_DIR}/ — run a model first.")
    else:
        ap.print_help()
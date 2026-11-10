"""Read-only check: do some rinks record shot locations differently?

A rink whose scorers log shots closer than they were inflates xG there, so goals / xG
comes in under 1; logging them farther out pushes it over 1. Recorded distance itself
moves too. But the home team plays half its games at its own rink, so a raw rink average
mixes up scorekeeping with the home team's style. So each rink is compared like with like.

METHOD
------
- Rinks are the games table's `venue`, regular season, 2022-23 to 2025-26, one season at
  a time. A venue that hosted fewer than 20 games that season isn't a rink here (outdoor and
  overseas games), but its shots still count in the teams' "other games".
- Shots: unblocked attempts in shot_features_fenwick, empty-net shots left out, xG from
  v1.4.1-fenwick.
- Comparison A, by shooter: every team that shot at the rink, against that team's own shots
  in its other games that season, weighted by how many shots it took at the rink.
  Comparison B, by defender: the same, using the team the shots were against. B is there
  because A alone would count a strong home goalie, or a home team that gives up close
  shots, as a rink effect. A rink is called off only when A and B agree.
- Ranges: 95% intervals from resampling the season's games (all of them together,
  so the comparison side is resampled too), 1,000 times.
- The median distance is shown as a point only. Means, shares and goals / xG get ranges.

Run: .venv/bin/python -m diagnostics.check_rinks > results/checks/rinks.txt
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ingest.db import pg_conn

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

MODEL = "v1.4.1-fenwick"
SEASONS = [20222023, 20232024, 20242025, 20252026]
MIN_GAMES = 20
BOOT = 1000
SEED = 2027

SQL = """
SELECT g.season, g.game_id, g.venue, g.home_team_abbr AS home,
       CASE WHEN s.shooter_team_id = g.home_team_id THEN g.home_team_abbr
            ELSE g.away_team_abbr END AS shooter,
       CASE WHEN s.shooter_team_id = g.home_team_id THEN g.away_team_abbr
            ELSE g.home_team_abbr END AS defender,
       f.distance_ft AS d, f.is_goal::int AS goal, x.xg,
       s.raw_details_json->>'shotType' = 'tip-in' AS tip
FROM shot_features_fenwick f
JOIN shots s ON s.shot_id = f.shot_id
JOIN shot_xg x ON x.shot_id = f.shot_id AND x.model_version = %(model)s
JOIN games g ON g.game_id = f.game_id
WHERE g.game_type = 2 AND g.season = ANY(%(seasons)s) AND NOT f.empty_net
"""

# Per-shot columns summed per (game, team); each rate is column / n.
RATES = {"mean_d": "d", "within_20": "c20", "within_10": "c10", "tip_share": "tip"}


def section(title: str, note: str = "") -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90 + (f"\n{note}\n" if note else ""))


def season_label(s: int) -> str:
    return f"{str(s)[:4]}-{str(s)[6:]}"


class Season:
    """One season's per-(game, team) sums, set up for fast weighted recomputation."""

    def __init__(self, shots: pd.DataFrame, side: str):
        shots = shots.assign(n=1, c20=shots.d.le(20).astype(int), c10=shots.d.le(10).astype(int),
                             tip=shots.tip.astype(int))
        cols = ["n", "d", "c20", "c10", "tip", "goal", "xg"]
        agg = shots.groupby(["game_id", "venue", side], as_index=False)[cols].sum()
        self.games = np.sort(agg.game_id.unique())
        self.g = np.searchsorted(self.games, agg.game_id.to_numpy())
        self.venues = np.sort(agg.venue.unique())
        self.teams = np.sort(agg[side].unique())
        v = np.searchsorted(self.venues, agg.venue.to_numpy())
        self.t = np.searchsorted(self.teams, agg[side].to_numpy())
        self.vt = v * len(self.teams) + self.t
        self.A = agg[cols].to_numpy(float)
        self.cols = cols

    def compute(self, w: np.ndarray) -> dict:
        """Rink vs matched comparison for every venue, with per-game weights w."""
        nv, nt = len(self.venues), len(self.teams)
        rw = w[self.g][:, None] * self.A
        S_vt = np.stack([np.bincount(self.vt, rw[:, j], nv * nt).reshape(nv, nt)
                         for j in range(rw.shape[1])], axis=-1)
        S_t = np.stack([np.bincount(self.t, rw[:, j], nt) for j in range(rw.shape[1])], axis=-1)
        N_vt, N_t = S_vt[..., 0], S_t[:, 0]
        other_n = N_t[None, :] - N_vt
        with np.errstate(divide="ignore", invalid="ignore"):
            other_rate = np.where(other_n[..., None] > 0,
                                  (S_t[None, :, :] - S_vt) / other_n[..., None], 0.0)
            at = S_vt.sum(axis=1) / N_vt.sum(axis=1)[:, None]
            base = (N_vt[..., None] * other_rate).sum(axis=1) / N_vt.sum(axis=1)[:, None]
        ix = {c: i for i, c in enumerate(self.cols)}
        out = {}
        for name, c in RATES.items():
            out[name] = at[:, ix[c]]
            out[name + "_base"] = base[:, ix[c]]
        out["oe"] = at[:, ix["goal"]] / at[:, ix["xg"]]
        out["oe_base"] = base[:, ix["goal"]] / base[:, ix["xg"]]
        out["shots"] = N_vt.sum(axis=1)
        return out


def weighted_median(values, weights) -> float:
    o = np.argsort(values)
    cw = np.cumsum(weights[o])
    return float(values[o][np.searchsorted(cw, cw[-1] / 2)])


def matched_medians(shots: pd.DataFrame, rinks, side: str) -> dict:
    """Rink median distance, and the median of the teams' other-game shots reweighted."""
    res = {}
    for venue in rinks:
        at = shots[shots.venue == venue]
        share = at[side].value_counts()
        other = shots[(shots.venue != venue) & shots[side].isin(share.index)]
        cnt = other[side].value_counts()
        w = (other[side].map(share) / other[side].map(cnt)).to_numpy()
        res[venue] = (float(at.d.median()), weighted_median(other.d.to_numpy(), w))
    return res


def run_season(shots: pd.DataFrame, rng: np.random.Generator):
    """Both comparisons for one season: point estimates and bootstrap draws."""
    games = shots.groupby("venue").game_id.nunique()
    home = shots.groupby("venue").home.agg(lambda s: s.mode().iloc[0])
    rinks = games[games >= MIN_GAMES].index
    out = {}
    idx = None
    for side in ("shooter", "defender"):
        S = Season(shots, side)
        if idx is None:
            idx = rng.integers(0, len(S.games), size=(BOOT, len(S.games)))
        point = S.compute(np.ones(len(S.games)))
        draws = [S.compute(np.bincount(i, minlength=len(S.games)).astype(float)) for i in idx]
        keep = np.isin(S.venues, rinks)
        venues = S.venues[keep]

        def eff(d):
            return {"mean_d": d["mean_d"] - d["mean_d_base"],
                    "within_20": d["within_20"] - d["within_20_base"],
                    "within_10": d["within_10"] - d["within_10_base"],
                    "tip_share": d["tip_share"] - d["tip_share_base"],
                    "oe": d["oe"], "oe_rel": d["oe"] / d["oe_base"]}

        p = eff(point)
        b = [eff(d) for d in draws]
        frame = pd.DataFrame({"venue": venues, "home": home.loc[venues].to_numpy(),
                              "games": games.loc[venues].to_numpy(),
                              "shots": point["shots"][keep].astype(int)})
        for k in p:
            arr = np.stack([d[k][keep] for d in b])
            frame[k] = p[k][keep]
            frame[k + "_lo"] = np.nanquantile(arr, 0.025, axis=0)
            frame[k + "_hi"] = np.nanquantile(arr, 0.975, axis=0)
            frame[k + "_draws"] = list(arr.T)
        frame["mean_d_at"] = point["mean_d"][keep]
        frame["mean_d_base"] = point["mean_d_base"][keep]
        meds = matched_medians(shots, venues, side)
        frame["median_d_at"] = [meds[v][0] for v in venues]
        frame["median_d_base"] = [meds[v][1] for v in venues]
        out[side] = frame
    return out


def rng_text(v, lo, hi, fmt):
    return f"{fmt.format(v)} ({fmt.format(lo)} to {fmt.format(hi)})"


def clear(r, k, null):
    return (r[k + "_lo"] > null) or (r[k + "_hi"] < null)


def main() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(SQL, conn, params={"model": MODEL, "seasons": SEASONS})
    print(f"Rink-by-rink shot location check, model {MODEL}, regular season "
          f"{season_label(SEASONS[0])} to {season_label(SEASONS[-1])}.")
    print(f"{len(df):,} unblocked, non-empty-net attempts in {df.game_id.nunique():,} games. "
          f"Rinks: venues with {MIN_GAMES}+ games in the season. Ranges: 95%, "
          f"{BOOT} game resamples.")
    print("A = against the shooting teams' shots in their other games; "
          "B = against the defending teams' other games.")

    rng = np.random.default_rng(SEED)
    res = {s: run_season(df[df.season == s], rng) for s in SEASONS}

    # -------------------------------------------------------------- per season
    for s in SEASONS:
        a, b = res[s]["shooter"], res[s]["defender"].set_index("venue")
        section(f"1. {season_label(s)}, rink vs. matched (A), sorted by mean distance gap",
                "d gap = rink mean distance minus matched, ft. <=20 gap = share within 20 ft, "
                "points.\nO/E = goals / xG at the rink. rel = rink O/E / matched O/E. "
                "* = range clear of 0 (or 1) in both A and B.")
        rows = []
        for _, r in a.sort_values("mean_d").iterrows():
            rb = b.loc[r.venue]
            star = lambda k, null: "*" if clear(r, k, null) and clear(rb, k, null) else " "  # noqa: E731
            rows.append({
                "rink": f"{r.home} {r.venue}"[:30], "games": r.games, "shots": r.shots,
                "mean d": f"{r.mean_d_at:.1f} vs {r.mean_d_base:.1f}",
                "d gap A": rng_text(r.mean_d, r.mean_d_lo, r.mean_d_hi, "{:+.1f}") + star("mean_d", 0),
                "d gap B": f"{rb.mean_d:+.1f}",
                "median": f"{r.median_d_at:.0f} vs {r.median_d_base:.0f}",
                "<=20 gap A": rng_text(r.within_20 * 100, r.within_20_lo * 100,
                                       r.within_20_hi * 100, "{:+.1f}") + star("within_20", 0),
                "O/E": rng_text(r.oe, r.oe_lo, r.oe_hi, "{:.2f}") + star("oe", 1),
                "rel A": rng_text(r.oe_rel, r.oe_rel_lo, r.oe_rel_hi, "{:.2f}") + star("oe_rel", 1),
                "rel B": f"{rb.oe_rel:.2f}",
            })
        print(pd.DataFrame(rows).to_string(index=False))

    # -------------------------------------------------------------- noise check
    section("2. Is there more spread between rinks than chance would give?",
            "z = gap / its resampling spread. With no rink effects at all, z averages 0 and its\n"
            "variance is about 1; a variance well above 1 means rinks really differ. 'clear' =\n"
            "rinks whose range clears 0 (or 1) in both A and B; about 5% would by chance alone.")
    rows = []
    for s in SEASONS:
        a, b = res[s]["shooter"], res[s]["defender"].set_index("venue")
        for k, null in (("mean_d", 0), ("within_20", 0), ("oe", 1), ("oe_rel", 1)):
            sd = np.array([np.nanstd(x) for x in a[k + "_draws"]])
            z = (a[k] - null) / sd
            both = sum(clear(r, k, null) and clear(b.loc[r.venue], k, null) for _, r in a.iterrows())
            rows.append({"season": season_label(s), "measure": k, "rinks": len(a),
                         "var(z)": round(float(np.var(z)), 2), "clear": both})
    print(pd.DataFrame(rows).pivot(index="measure", columns="season",
                                   values=["var(z)", "clear"]).to_string())

    # -------------------------------------------------------------- consistency
    section("3. Rinks off in the same direction every season (comparison A)",
            "Sign of the gap in each season (+ farther / higher, - closer / lower); "
            "'clear' = seasons where\nthe range clears 0 (or 1) in both A and B. "
            "4 seasons: pooled gap, its range from the season resamples\n"
            "(weighted by shots). By chance alone, about 1 rink in 8 has the same sign "
            "all 4 seasons.")
    keyed = {}
    for s in SEASONS:
        a, b = res[s]["shooter"], res[s]["defender"].set_index("venue")
        for _, r in a.iterrows():
            keyed.setdefault(r.home, {})[s] = (r, b.loc[r.venue])
    for k, null, fmt, scale in (("mean_d", 0, "{:+.2f}", 1), ("within_20", 0, "{:+.2f}", 100),
                                ("oe", 1, "{:.3f}", 1), ("oe_rel", 1, "{:.3f}", 1)):
        rows = []
        for home, by in keyed.items():
            if len(by) < len(SEASONS):
                continue
            vals = [by[s][0][k] for s in SEASONS]
            signs = "".join("+" if v > null else "-" for v in vals)
            nclear = sum(clear(by[s][0], k, null) and clear(by[s][1], k, null) for s in SEASONS)
            wts = np.array([by[s][0].shots for s in SEASONS], float)
            draws = np.stack([by[s][0][k + "_draws"] for s in SEASONS])
            pooled = np.average(vals, weights=wts)
            pd_draws = np.nansum(draws * wts[:, None], axis=0) / wts.sum()
            lo, hi = np.nanquantile(pd_draws, [0.025, 0.975])
            rows.append({"home": home, "venues": " / ".join(sorted({by[s][0].venue for s in SEASONS})),
                         "signs": signs, "clear": nclear,
                         "pooled": rng_text(pooled * scale, lo * scale, hi * scale, fmt),
                         "_same": len(set(signs)) == 1, "_p": pooled,
                         "_clr": (lo > null) or (hi < null)})
        t = pd.DataFrame(rows)
        same = t[t._same].sort_values("_p")
        print(f"\n--- {k}: {len(same)} of {len(t)} rinks same sign all 4 seasons "
              f"(chance: about {len(t) / 8:.0f}); pooled range clear of {null}: {int(t._clr.sum())} ---")
        print(same.drop(columns=["_same", "_p", "_clr"]).to_string(index=False))

    # -------------------------------------------------------------- Anaheim
    section("4. Anaheim: are the extra close tip-ins part of a wider close-shot pattern?",
            "Honda Center vs. matched (A and B), by season. Tip-ins and all other shots "
            "separately.\nwithin 10 / within 20 in points; tip share = tip-ins as a share of all "
            "attempts.")
    rows = []
    for s in SEASONS:
        sub = df[df.season == s]
        venue = res[s]["shooter"].set_index("home").loc["ANA", "venue"]
        for label, part in (("tip-ins", sub[sub.tip]), ("all other shots", sub[~sub.tip]),
                            ("all shots", sub)):
            for side in ("shooter", "defender"):
                S = Season(part, side)
                p = S.compute(np.ones(len(S.games)))
                i = int(np.searchsorted(S.venues, venue))
                draws = []
                for idx in rng.integers(0, len(S.games), size=(300, len(S.games))):
                    d = S.compute(np.bincount(idx, minlength=len(S.games)).astype(float))
                    draws.append((d["within_10"][i] - d["within_10_base"][i],
                                  d["mean_d"][i] - d["mean_d_base"][i],
                                  d["tip_share"][i] - d["tip_share_base"][i]))
                q = np.nanquantile(np.array(draws), [0.025, 0.975], axis=0)
                rows.append({
                    "season": season_label(s), "shots": label, "vs": "A" if side == "shooter" else "B",
                    "n": int(p["shots"][i]),
                    "within 10": f"{p['within_10'][i] * 100:.1f} vs {p['within_10_base'][i] * 100:.1f}",
                    "gap": rng_text((p["within_10"][i] - p["within_10_base"][i]) * 100,
                                    q[0, 0] * 100, q[1, 0] * 100, "{:+.1f}"),
                    "mean d gap": rng_text(p["mean_d"][i] - p["mean_d_base"][i], q[0, 1], q[1, 1],
                                           "{:+.1f}"),
                    "tip share gap": (rng_text((p["tip_share"][i] - p["tip_share_base"][i]) * 100,
                                               q[0, 2] * 100, q[1, 2] * 100, "{:+.1f}")
                                      if label == "all shots" else ""),
                })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

# elo_predictor.py  — no Streamlit imports, tolerant CSV schema, shared fetchers
import datetime as dt
import math
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import requests

# ---------------------------
# ELO constants
# ---------------------------
START_ELO = 1500
K = 20
HOME_FIELD_BONUS = 65  # ~2 NFL points

# ---------------------------
# ESPN helpers (Regular Season = seasontype 2)
# ---------------------------
ESPN_SB_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def _espn_params(year: int, week: Optional[int]) -> Dict[str, int]:
    """
    Build ESPN scoreboard params.
    - seasontype=2 => Regular season
    - dates=YEAR   => Scope to that NFL season
    - omit 'week'  => ESPN returns the current week
    """
    base = {"seasontype": 2, "dates": year}
    if week is not None:
        base["week"] = week
    return base

def get_current_week(year: int) -> Optional[int]:
    """Return the current NFL week number for the season, or None if unavailable."""
    try:
        r = requests.get(ESPN_SB_URL, params=_espn_params(year, None), timeout=15)
        r.raise_for_status()
        data = r.json()
        wk = (data.get("week") or {}).get("number")
        if isinstance(wk, int) and 1 <= wk <= 23:
            return wk
    except Exception:
        pass
    return None

def fetch_schedule(year: int, week: int) -> pd.DataFrame:
    """
    Fetch a week's schedule/results from ESPN, robustly detecting home/away.
    Returns columns: date, home_team, away_team, home_score, away_score, status, source
    """
    try:
        data = requests.get(ESPN_SB_URL, params=_espn_params(year, week), timeout=20).json()
    except Exception:
        data = {}

    rows: List[Dict] = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]

        home = away = None
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        if not home or not away:
            comps = comp.get("competitors", [])
            if len(comps) >= 2:
                home, away = comps[0], comps[1]

        def name(c):
            return (c.get("team") or {}).get("displayName") if c else None

        def score(c):
            s = c.get("score") if c else None
            try:
                return int(s) if s not in (None, "") else None
            except Exception:
                return None

        rows.append({
            "date": comp.get("date") or event.get("date"),
            "home_team": name(home),
            "away_team": name(away),
            "home_score": score(home),
            "away_score": score(away),
            "status": (comp.get("status") or {}).get("type", {}).get("description"),
            "source": "ESPN",
        })

    df = pd.DataFrame(rows, columns=["date","home_team","away_team","home_score","away_score","status","source"])

    # Optional: if ESPN returns empty (rare), you could add a PFR fallback here.
    return df

# ---------------------------
# ELO math
# ---------------------------
def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

def update_elo(elo_a: float, elo_b: float, score_a: int, score_b: int) -> (float, float):
    exp_a = expected_score(elo_a, elo_b)
    if score_a > score_b:
        result_a = 1.0
    elif score_a < score_b:
        result_a = 0.0
    else:
        result_a = 0.5
    result_b = 1.0 - result_a
    new_a = elo_a + K * (result_a - exp_a)
    new_b = elo_b + K * (result_b - (1.0 - exp_a))
    return new_a, new_b

def build_elos(csv_path: str = "data/historical_results.csv") -> Dict[str, float]:
    """
    Build ELOs from a historical results CSV, tolerating different schemas:
      • PFR game logs:      Winner/tie, Loser/tie, Pts, Pts.1
      • Alternate exports:  Winner/tie, Loser/tie, Points_Winner/Points_Loser or PtsW/PtsL
      • Custom schedules:   home_team, away_team, home_score, away_score
    """
    df = pd.read_csv(csv_path)
    elos: Dict[str, float] = {}

    # Normalize column lookup (case-insensitive)
    colmap = {str(c).strip().lower(): c for c in df.columns}
    def has(*names): return all(n.lower() in colmap for n in names)
    def col(name):  return colmap.get(name.lower())

    def as_int(x):
        v = pd.to_numeric(x, errors="coerce")
        return int(v) if pd.notna(v) and not math.isnan(v) else None

    def row_to_result(row):
        # Case A: PFR-style with Winner/Loser
        if has("winner/tie", "loser/tie"):
            winner = row[col("Winner/tie")]
            loser  = row[col("Loser/tie")]
            for wcol, lcol in [("Points_Winner","Points_Loser"), ("PtsW","PtsL"), ("Pts","Pts.1")]:
                if has(wcol, lcol):
                    sw = as_int(row[col(wcol)])
                    sl = as_int(row[col(lcol)])
                    if sw is not None and sl is not None:
                        return winner, loser, sw, sl
            # If scores absent, at least count a win/loss (minimal)
            return winner, loser, 1, 0

        # Case B: home/away with explicit scores
        if has("home_team", "away_team"):
            home, away = row[col("home_team")], row[col("away_team")]
            hs = as_int(row[col("home_score")]) if has("home_score") else None
            as_ = as_int(row[col("away_score")]) if has("away_score") else None
            if hs is None or as_ is None:
                return None
            if hs > as_:
                return home, away, hs, as_
            elif as_ > hs:
                return away, home, as_, hs
            else:
                return home, away, hs, as_

        return None

    for _, row in df.iterrows():
        parsed = row_to_result(row)
        if not parsed:
            continue
        winner, loser, sw, sl = parsed
        elos.setdefault(winner, START_ELO)
        elos.setdefault(loser, START_ELO)
        elos[winner], elos[loser] = update_elo(elos[winner], elos[loser], sw, sl)

    return elos

# ---------------------------
# Team stats + prediction
# ---------------------------
def get_team_stats(year: Optional[int] = None) -> pd.DataFrame:
    """Simple team table from PFR main season page."""
    year = year or dt.datetime.now().year
    url = f"https://www.pro-football-reference.com/years/{year}/"
    tables = pd.read_html(url)
    stats = tables[0]

    # Guard against missing columns early in season
    for c in ("TO", "Yds"):
        if c not in stats.columns:
            stats[c] = 0

    stats = stats.rename(columns={
        "Tm": "Team",
        "PF": "Points_For",
        "PA": "Points_Against",
        "TO": "Turnovers",
        "Yds": "Yards",
    })
    stats = stats[["Team", "Points_For", "Points_Against", "Turnovers", "Yards"]]
    for c in ["Points_For", "Points_Against", "Turnovers", "Yards"]:
        stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0)
    stats["Point_Diff"] = stats["Points_For"] - stats["Points_Against"]
    return stats.set_index("Team")

def predict_matchup(team_home: str, team_away: str, elos: Dict[str, float], stats: pd.DataFrame, home_team: Optional[str] = None) -> Dict[str, float]:
    # ELO with home-field
    elo_home = elos.get(team_home, START_ELO)
    elo_away = elos.get(team_away, START_ELO)
    if home_team == team_home:
        elo_home += HOME_FIELD_BONUS
    elif home_team == team_away:
        elo_away += HOME_FIELD_BONUS
    elo_prob_home = expected_score(elo_home, elo_away)

    # Stats blend
    s_home = stats.loc[team_home] if team_home in stats.index else None
    s_away = stats.loc[team_away] if team_away in stats.index else None
    point_diff_factor = 0.5
    turnover_factor = 0.5
    if s_home is not None and s_away is not None:
        pd_home, pd_away = s_home["Point_Diff"], s_away["Point_Diff"]
        point_diff_factor = (pd_home - pd_away) / 200.0 + 0.5
        to_home, to_away = -s_home["Turnovers"], -s_away["Turnovers"]  # fewer turnovers -> better
        turnover_factor = (to_home - to_away) / 50.0 + 0.5

    final_prob_home = float(np.clip(0.6 * elo_prob_home + 0.25 * point_diff_factor + 0.15 * turnover_factor, 0, 1))
    return {
        "team_home": team_home,
        "team_away": team_away,
        "elo_prob_home": float(elo_prob_home),
        "final_prob": final_prob_home,  # probability the HOME team wins
    }

# ---------------------------
# CLI (optional)
# ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--hist", type=str, default="data/historical_results.csv")
    args = ap.parse_args()

    Y = args.year or dt.datetime.now().year
    W = args.week or (get_current_week(Y) or 1)
    sched = fetch_schedule(Y, W)
    if sched.empty:
        print(f"No games found for {Y} Week {W}.")
        raise SystemExit(0)

    elos_ = build_elos(args.hist)
    stats_ = get_team_stats(Y)

    print(f"\n📊 {Y} Week {W} Predictions\n")
    for _, g in sched.iterrows():
        home, away = g["home_team"], g["away_team"]
        res = predict_matchup(home, away, elos_, stats_, home_team=home)
        p_home = res["final_prob"]
        print(f"{away} @ {home}  —  {home}: {p_home:.2%},  {away}: {(1-p_home):.2%}")


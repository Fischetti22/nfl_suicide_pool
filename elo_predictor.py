import pandas as pd
import numpy as np
import requests

START_ELO = 1500
K = 20
HOME_FIELD_BONUS = 65  # ~2 points of NFL advantage

# ---------------------------
# ELO FUNCTIONS
# ---------------------------
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, score_a, score_b):
    exp_a = expected_score(elo_a, elo_b)
    exp_b = 1 - exp_a

    if score_a > score_b:
        result_a, result_b = 1, 0
    elif score_a < score_b:
        result_a, result_b = 0, 1
    else:
        result_a, result_b = 0.5, 0.5

    new_elo_a = elo_a + K * (result_a - exp_a)
    new_elo_b = elo_b + K * (result_b - exp_b)
    return new_elo_a, new_elo_b

def build_elos(csv_path="data/historical_results.csv"):
    """
    Build ELOs from a historical results CSV, tolerating different schemas:
      • PFR game logs:      Winner/tie, Loser/tie, Pts, Pts.1
      • Some exports:       Winner/tie, Loser/tie, Points_Winner, Points_Loser (or PtsW/PtsL)
      • Custom schedules:   home_team, away_team, home_score, away_score
    """
    import math
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols_norm = {str(c).strip().lower(): c for c in df.columns}

    def has(*names):
        return all(n.lower() in cols_norm for n in names)

    def col(name):
        return cols_norm.get(name.lower())

    elos = {}

    def as_int(x):
        v = pd.to_numeric(x, errors="coerce")
        return int(v) if pd.notna(v) and not math.isnan(v) else None

    def row_to_result(row):
        # Case A: “Winner/tie / Loser/tie” schema (PFR-style)
        if has("winner/tie", "loser/tie"):
            winner = row[col("Winner/tie")]
            loser  = row[col("Loser/tie")]

            # Try several common score-pair column names
            for wcol, lcol in [
                ("Points_Winner", "Points_Loser"),
                ("PtsW", "PtsL"),
                ("Pts", "Pts.1"),
            ]:
                if has(wcol, lcol):
                    sw = as_int(row[col(wcol)])
                    sl = as_int(row[col(lcol)])
                    if sw is not None and sl is not None:
                        return winner, loser, sw, sl

            # No scores found? Still count a win/loss minimally
            return winner, loser, 1, 0

        # Case B: home/away schema
        if has("home_team", "away_team"):
            home = row[col("home_team")]
            away = row[col("away_team")]
            hs = as_int(row[col("home_score")]) if has("home_score") else None
            as_ = as_int(row[col("away_score")]) if has("away_score") else None

            # If scores missing, skip (can’t determine winner)
            if hs is None or as_ is None:
                return None

            if hs > as_:
                return home, away, hs, as_
            elif as_ > hs:
                return away, home, as_, hs
            else:
                # tie
                return home, away, hs, as_

        # Unknown schema → skip
        return None

    for _, row in df.iterrows():
        parsed = row_to_result(row)
        if not parsed:
            continue
        winner, loser, sw, sl = parsed

        elos.setdefault(winner, START_ELO)
        elos.setdefault(loser, START_ELO)
        new_w, new_l = update_elo(elos[winner], elos[loser], sw, sl)
        elos[winner], elos[loser] = new_w, new_l

    return elos



# ---------------------------
# TEAM STATS SCRAPER
# ---------------------------
def get_team_stats(year=2024):
    url = f"https://www.pro-football-reference.com/years/{year}/"
    tables = pd.read_html(url)
    stats = tables[0]

    # Ensure 'TO' and 'Yds' columns exist before renaming
    if 'TO' not in stats.columns:
        stats['TO'] = 0
    if 'Yds' not in stats.columns:
        stats['Yds'] = 0

    stats = stats.rename(columns={
        "Tm": "Team",
        "PF": "Points_For",
        "PA": "Points_Against",
        "TO": "Turnovers",
        "Yds": "Yards"
    })

    stats = stats[["Team", "Points_For", "Points_Against", "Turnovers", "Yards"]]
    stats["Points_For"] = pd.to_numeric(stats["Points_For"], errors='coerce')
    stats["Points_Against"] = pd.to_numeric(stats["Points_Against"], errors='coerce')
    stats["Point_Diff"] = stats["Points_For"] - stats["Points_Against"]
    return stats.set_index("Team")

# ---------------------------
# HYBRID PREDICTION
# ---------------------------
def predict_matchup(team_a, team_b, elos, stats, home_team=None):
    # ELO base
    elo_a = elos.get(team_a, START_ELO)
    elo_b = elos.get(team_b, START_ELO)

    if home_team == team_a:
        elo_a += HOME_FIELD_BONUS
    elif home_team == team_b:
        elo_b += HOME_FIELD_BONUS

    elo_prob = expected_score(elo_a, elo_b)

    # Team stats factors
    stat_a = stats.loc[team_a] if team_a in stats.index else None
    stat_b = stats.loc[team_b] if team_b in stats.index else None

    point_diff_factor = 0.5
    turnover_factor = 0.5

    if stat_a is not None and stat_b is not None:
        pd_a = stat_a["Point_Diff"]
        pd_b = stat_b["Point_Diff"]
        point_diff_factor = (pd_a - pd_b) / 200.0 + 0.5  # normalize

        to_a = -stat_a["Turnovers"]  # fewer TO = better
        to_b = -stat_b["Turnovers"]
        turnover_factor = (to_a - to_b) / 50.0 + 0.5

    # Blend
    final_prob = (
        0.6 * elo_prob +
        0.25 * point_diff_factor +
        0.15 * turnover_factor
    )

    final_prob = np.clip(final_prob, 0, 1)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "elo_prob": elo_prob,
        "point_diff_factor": point_diff_factor,
        "turnover_factor": turnover_factor,
        "final_prob": final_prob
    }

# ---------------------------
# MAIN INTERACTIVE LOOP
# ---------------------------
def predict_week(csv_path="data/current_results.csv",
                 hist_path="data/historical_results.csv",
                 year=2024):
    elos = build_elos(hist_path)
    stats = get_team_stats(year)
    current = pd.read_csv(csv_path)

    print("\n📊 Weekly Suicide Pool Predictions\n")
    for _, game in current.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        result = predict_matchup(home, away, elos, stats, home_team=home)
        prob_home = result["final_prob"]
        prob_away = 1 - prob_home

        print(f"{away} @ {home}")
        print(f"   {home} win probability: {prob_home:.2%}")
        print(f"   {away} win probability: {prob_away:.2%}\n")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    predict_week()


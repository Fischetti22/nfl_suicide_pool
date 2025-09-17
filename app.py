import streamlit as st
import pandas as pd
import elo_predictor as _ep
st.caption(f"ELO module path: {_ep.__file__}")


# ---------------------------
# FETCH MATCHUPS (ESPN -> fallback PFR)
# ---------------------------
def fetch_from_espn(year, week):
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {"season": year, "week": week}
    resp = requests.get(url, params=params).json()
    games = []
    for event in resp.get("events", []):
        comp = event["competitions"][0]
        home = comp["competitors"][0]
        away = comp["competitors"][1]
        games.append({
            "date": event["date"],
            "home_team": home["team"]["displayName"],
            "home_score": home["score"],
            "away_team": away["team"]["displayName"],
            "away_score": away["score"],
            "status": comp["status"]["type"]["description"],
            "source": "ESPN"
        })
    return games

def fetch_from_pfr(year, week):
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
    df = pd.read_html(url)[0]
    df = df[df["Week"].apply(lambda x: str(x).isdigit())]
    week_games = df[df["Week"].astype(int) == week]
    games = []
    for _, row in week_games.iterrows():
        if "@" in str(row.get("Unnamed: 5", "")):
            home_team = row["Loser/tie"]
            away_team = row["Winner/tie"]
        else:
            home_team = row["Winner/tie"]
            away_team = row["Loser/tie"]
        games.append({
            "date": row["Date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": None,
            "away_score": None,
            "status": "Scheduled",
            "source": "PFR"
        })
    return games

def fetch_current_results(year=2025, week=3):
    games = fetch_from_espn(year, week)
    if not games:
        games = fetch_from_pfr(year, week)
    return pd.DataFrame(games)

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("NFL ELO Predictor 🏈")

# Build ratings
# --- robust ELO loader (app-safe) ---
import os
from pathlib import Path
import elo_predictor as _ep

# See which module file the app is importing (helps catch stale deployments)
st.caption(f"ELO module path: {_ep.__file__}")
st.caption(f"CWD: {os.getcwd()}")

# Try a few likely CSV locations (works on Streamlit Cloud, Docker, local)
candidates = [
    Path("data/historical_results.csv"),
    Path(__file__).parent / "data" / "historical_results.csv",
    Path.cwd() / "data" / "historical_results.csv",
]
csv_path = next((p for p in candidates if p.exists()), None)
if not csv_path:
    st.error("Could not find data/historical_results.csv. Put it in a /data folder next to app.py.")
    st.stop()

# Show the columns so we know the schema the app sees
try:
    _cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
    st.caption(f"historical_results.csv columns: {_cols}")
except Exception as e:
    st.error(f"Failed to open {csv_path}: {e}")
    st.stop()

# Build ELOs (if this raises, show the traceback in the UI)
try:
    elos = build_elos(str(csv_path))
except Exception as e:
    st.error(f"build_elos() failed on {csv_path.name}: {e}")
    st.exception(e)
    st.stop()

stats = get_team_stats(2025)

# Auto-fetch current matchups
week = 3
year = 2025
current_df = fetch_current_results(year=year, week=week)

if not current_df.empty:
    st.subheader(f"📅 Week {week} Matchups ({year})")
    st.dataframe(current_df[["date", "home_team", "away_team", "status", "source"]])
else:
    st.warning("No games found for this week. Try later.")

# Select game from schedule
if not current_df.empty:
    games = current_df.apply(lambda x: f"{x['away_team']} @ {x['home_team']}", axis=1).tolist()
    game_choice = st.selectbox("Choose a game", games)
    row = current_df.iloc[games.index(game_choice)]
    team_a = row["home_team"]
    team_b = row["away_team"]
    home_team = team_a
else:
    # fallback manual mode
    team_a = st.selectbox("Select Team A", list(elos.keys()))
    team_b = st.selectbox("Select Team B", list(elos.keys()))
    home_team = st.selectbox("Home Team (optional)", ["None"] + list(elos.keys()))
    home_team = None if home_team == "None" else home_team

# Single prediction
if st.button("Predict Selected Game"):
    result = predict_matchup(team_a, team_b, elos, stats, home_team=home_team)
    st.write(f"**{team_a} win probability:** {result['final_prob']:.2%}")
    st.write(f"**{team_b} win probability:** {1-result['final_prob']:.2%}")

# ---------------------------
# RANKED SAFEST PICKS
# ---------------------------
if not current_df.empty:
    st.subheader("🔒 Safest Picks This Week")

    ranked = []
    for _, row in current_df.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        result = predict_matchup(team_a, team_b, elos, stats, home_team=team_a)

        ranked.append({
            "Matchup": f"{team_b} @ {team_a}",
            "Favorite": team_a if result["final_prob"] >= 0.5 else team_b,
            "Favorite Win %": max(result["final_prob"], 1 - result["final_prob"]),
            "Underdog": team_b if result["final_prob"] >= 0.5 else team_a,
            "Underdog Win %": min(result["final_prob"], 1 - result["final_prob"])
        })

    ranked_df = pd.DataFrame(ranked).sort_values("Favorite Win %", ascending=False)
    st.dataframe(ranked_df)


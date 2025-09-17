import streamlit as st
import pandas as pd
from elo_predictor import build_elos, get_team_stats, predict_matchup
import requests

# ---------------------------
# FETCH MATCHUPS (ESPN -> fallback PFR)
# ---------------------------
def fetch_from_espn(year, week):
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {"year": year, "week": week, "seasontype": 2}
    resp = requests.get(url, params=params, timeout=20).json()
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

def current_year_week():
    now = pd.Timestamp.now()
    year = now.year
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        j = requests.get(url, params={"year": year, "seasontype": 2}, timeout=20).json()
        wk = (j.get("week") or {}).get("number")
        if isinstance(wk, int) and 1 <= wk <= 23:
            return year, wk
    except Exception:
        pass
    return year, 1


def fetch_current_results(year=None, week=None):
    if year is None or week is None:
        y, w = current_year_week()
        year = y if year is None else year
        week = w if week is None else week
    games = fetch_from_espn(year, week)
    if not games:
        games = fetch_from_pfr(year, week)
    return pd.DataFrame(games)

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("NFL ELO Predictor 🏈")

# Build ratings
elos = build_elos("data/historical_results.csv")
stats = get_team_stats(year)

# Auto-fetch current matchups
year, week = current_year_week()
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


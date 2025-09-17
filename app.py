# app.py — current-week, unified with elo_predictor
import os
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
import elo_predictor as _ep
from elo_predictor import (
    build_elos,
    get_team_stats,
    predict_matchup,
    fetch_schedule,
    get_current_week,
)

st.set_page_config(page_title="NFL ELO Predictor — Current Week", page_icon="🏈", layout="wide")
st.title("🏈 NFL ELO Predictor — Current Week")

# --- diagnostic: confirm which module/file is live ---
st.caption(f"ELO module path: {_ep.__file__}")
st.caption(f"CWD: {os.getcwd()}")

# --- detect current season & week (auto; no widgets) ---
YEAR = dt.datetime.now().year
WEEK = get_current_week(YEAR) or 1
st.markdown(f"**Detected:** {YEAR} — **Week {WEEK}** (auto)")

# --- locate historical_results.csv robustly ---
candidates = [
    Path("data/historical_results.csv"),
    Path(__file__).parent / "data" / "historical_results.csv",
    Path.cwd() / "data" / "historical_results.csv",
]
csv_path = next((p for p in candidates if p.exists()), None)
if not csv_path:
    st.error("Could not find data/historical_results.csv. Put it in a /data folder next to app.py.")
    st.stop()

# show CSV schema (helps catch column mismatches)
try:
    _cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
    st.caption(f"historical_results.csv columns: {_cols}")
except Exception as e:
    st.error(f"Failed to open {csv_path}: {e}")
    st.stop()

# --- build ELOs (tolerant build_elos in elo_predictor.py) ---
try:
    elos = build_elos(str(csv_path))
except Exception as e:
    st.error(f"build_elos() failed on {csv_path.name}: {e}")
    st.exception(e)
    st.stop()

# team stats for this season
try:
    stats = get_team_stats(YEAR)
except Exception as e:
    st.error(f"get_team_stats({YEAR}) failed: {e}")
    st.exception(e)
    st.stop()

# --- schedule: use shared fetch_schedule (no duplicate fetchers here) ---
schedule_df = fetch_schedule(YEAR, WEEK)
if schedule_df.empty:
    st.warning("No games found for the current week yet. Hard refresh the browser or check back later.")
    st.stop()

with st.expander("📅 Weekly schedule", expanded=True):
    view_cols = [c for c in ["date", "away_team", "home_team", "away_score", "home_score", "status", "source"] if c in schedule_df.columns]
    st.dataframe(schedule_df[view_cols], use_container_width=True)

# --- single-game prediction from the schedule ---
games = schedule_df.apply(lambda x: f"{x['away_team']} @ {x['home_team']}", axis=1).tolist()
choice = st.selectbox("Choose a game", games)
row = schedule_df.iloc[games.index(choice)]
home_team = row["home_team"]
away_team = row["away_team"]

if st.button("Predict selected game"):
    res = predict_matchup(home_team, away_team, elos, stats, home_team=home_team)
    p_home = float(res["final_prob"])
    st.write(f"**{home_team} win probability:** {p_home:.2%}")
    st.write(f"**{away_team} win probability:** {(1 - p_home):.2%}")

# --- ranked safest picks ---
st.subheader("🔒 Safest Picks This Week")
ranked = []
for _, g in schedule_df.iterrows():
    h, a = g["home_team"], g["away_team"]
    r = predict_matchup(h, a, elos, stats, home_team=h)
    p_home = float(r["final_prob"])
    favorite = h if p_home >= 0.5 else a
    favp = max(p_home, 1 - p_home)
    underdog = a if favorite == h else h
    ranked.append({
        "Matchup": f"{a} @ {h}",
        "Favorite": favorite,
        "Favorite Win %": round(100 * favp, 2),
        "Underdog": underdog,
        "Underdog Win %": round(100 * (1 - favp), 2),
    })
ranked_df = pd.DataFrame(ranked).sort_values("Favorite Win %", ascending=False, ignore_index=True)
st.dataframe(ranked_df, use_container_width=True)

# manual rerun (no caching in this file)
if st.button("🔄 Rerun now"):
    st.experimental_rerun()


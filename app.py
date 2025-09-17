# app.py — current week only, unified with elo_predictor, no duplicates
import os
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

# Safe import with checks so errors show in the UI
try:
    import elo_predictor as ep
except Exception as e:
    st.error("Failed to import elo_predictor.")
    st.exception(e)
    st.stop()

# Ensure required functions exist
required = ["build_elos", "get_team_stats", "predict_matchup", "fetch_schedule", "get_current_week"]
missing = [name for name in required if not hasattr(ep, name)]
if missing:
    st.error(f"`elo_predictor` is missing: {missing}. Make sure you deployed the updated file.")
    st.stop()

build_elos      = ep.build_elos
get_team_stats  = ep.get_team_stats
predict_matchup = ep.predict_matchup
fetch_schedule  = ep.fetch_schedule
get_current_week= ep.get_current_week

st.set_page_config(page_title="NFL Picks — Current Week", page_icon="🏈", layout="wide")
st.title("🏈 NFL Picks — Current Week")
st.caption(f"ELO module path: {ep.__file__}")
st.caption(f"CWD: {os.getcwd()}")

# Detect current week (no widgets)
YEAR = dt.datetime.now().year
WEEK = get_current_week(YEAR) or 1
st.markdown(f"**Detected:** {YEAR} — **Week {WEEK}**")

# Find historical_results.csv robustly
candidates = [
    Path("data/historical_results.csv"),
    Path(__file__).parent / "data" / "historical_results.csv",
    Path.cwd() / "data" / "historical_results.csv",
]
csv_path = next((p for p in candidates if p.exists()), None)
if not csv_path:
    st.error("Could not find data/historical_results.csv. Put it in a /data folder next to app.py.")
    st.stop()

# Show CSV schema
try:
    cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
    st.caption(f"historical_results.csv columns: {cols}")
except Exception as e:
    st.error(f"Failed to open {csv_path}: {e}")
    st.exception(e)
    st.stop()

# Build ELOs (tolerant)
try:
    elos = build_elos(str(csv_path))
except Exception as e:
    st.error(f"build_elos() failed on {csv_path.name}")
    st.exception(e)
    st.stop()

# Team stats for the season
try:
    stats = get_team_stats(YEAR)
except Exception as e:
    st.error(f"get_team_stats({YEAR}) failed")
    st.exception(e)
    st.stop()

# Fetch schedule using shared fetcher
schedule_df = fetch_schedule(YEAR, WEEK)
if schedule_df.empty:
    st.warning("No games found for the current week yet. Try again later.")
    st.stop()

with st.expander("📅 Weekly schedule", expanded=True):
    view_cols = [c for c in ["date","away_team","home_team","away_score","home_score","status","source"] if c in schedule_df.columns]
    st.dataframe(schedule_df[view_cols], use_container_width=True)

# One-game prediction
games = schedule_df.apply(lambda x: f"{x['away_team']} @ {x['home_team']}", axis=1).tolist()
choice = st.selectbox("Choose a game", games)
row = schedule_df.iloc[games.index(choice)]
home_team = row["home_team"]
away_team = row["away_team"]

if st.button("Predict selected game"):
    res = predict_matchup(home_team, away_team, elos, stats, home_team=home_team)
    p_home = float(res["final_prob"])
    st.write(f"**{home_team} win probability:** {p_home:.2%}")
    st.write(f"**{away_team} win probability:** {(1-p_home):.2%}")

# Ranked safest picks
st.subheader("🔒 Safest Picks This Week")
ranked_rows = []
for _, g in schedule_df.iterrows():
    h, a = g["home_team"], g["away_team"]
    r = predict_matchup(h, a, elos, stats, home_team=h)
    p_home = float(r["final_prob"])
    favorite = h if p_home >= 0.5 else a
    favp = max(p_home, 1 - p_home)
    underdog = a if favorite == h else h
    ranked_rows.append({
        "Matchup": f"{a} @ {h}",
        "Favorite": favorite,
        "Favorite Win %": round(100 * favp, 2),
        "Underdog": underdog,
        "Underdog Win %": round(100 * (1 - favp), 2),
    })
st.dataframe(pd.DataFrame(ranked_rows).sort_values("Favorite Win %", ascending=False, ignore_index=True),
             use_container_width=True)
             
# --- detect week and auto-advance if the whole week is Final ---
YEAR = dt.datetime.now().year
WEEK = get_current_week(YEAR) or 1

# fetch detected week
schedule_df = fetch_schedule(YEAR, WEEK)

def _all_final(df):
    if df.empty or "status" not in df.columns:
        return False
    s = df["status"].astype(str).str.lower()
    return s.notna().all() and (s == "final").all()

# if everything in the detected week is Final, try week+1
if _all_final(schedule_df) and WEEK < 23:
    next_df = fetch_schedule(YEAR, WEEK + 1)
    if not next_df.empty:
        WEEK += 1
        schedule_df = next_df

st.markdown(f"**Detected:** {YEAR} — **Week {WEEK}**")


# Manual rerun
if st.button("🔄 Rerun now"):
    st.experimental_rerun()


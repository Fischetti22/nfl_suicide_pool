import pandas as pd
import requests
import os

def scrape_season(year):
    """Scrape a full NFL season from Pro-Football-Reference."""
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
    tables = pd.read_html(url)
    df = tables[0]  # first table is schedule
    df = df[df["Week"].apply(lambda x: str(x).isdigit())]  # filter real weeks
    df = df.rename(columns={
        "Pts": "Points_Winner",
        "Pts.1": "Points_Loser",
        "YdsW": "Yards_Winner",
        "YdsL": "Yards_Loser",
        "TOW": "Turnovers_Winner",
        "TOL": "Turnovers_Loser"
    })
    df["season"] = year
    return df

def scrape_historical(years=5, out_path="data/historical_results.csv"):
    current_year = pd.Timestamp.now().year
    frames = []
    for y in range(current_year - years, current_year):
        print(f"Fetching {y} season...")
        frames.append(scrape_season(y))

    df_all = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"✅ Saved historical results to {out_path}")

# ---------------------------
# Minimal add-ons: THIS WEEK
# ---------------------------

def _espn_current_week(year: int) -> int | None:
    """
    Ask ESPN for the current NFL week for the given season (regular season).
    """
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"seasontype": 2, "dates": year}  # regular season, scoped to season year
        data = requests.get(url, params=params, timeout=15).json()
        wk = (data.get("week") or {}).get("number")
        if isinstance(wk, int) and 1 <= wk <= 23:
            return wk
    except Exception:
        pass
    return None

def _scrape_week_from_pfr(year: int, week: int) -> pd.DataFrame:
    """
    Scrape a single NFL week from PFR and return rows with the same
    column names you use above (Points_Winner/Points_Loser, etc.).
    Only keep completed games (both scores present) so ELO won’t break.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/week_{week}.htm"
    tables = pd.read_html(url)
    # Find the table that has Winner/Loser columns
    df = next((t for t in tables if {"Winner/tie","Loser/tie"}.issubset(set(map(str, t.columns)))), None)
    if df is None:
        return pd.DataFrame()

    # Normalize scores to your expected names
    rename_map = {
        "Pts": "Points_Winner",
        "Pts.1": "Points_Loser",
        "YdsW": "Yards_Winner",
        "YdsL": "Yards_Loser",
        "TOW": "Turnovers_Winner",
        "TOL": "Turnovers_Loser",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Keep only completed games (both scores exist)
    if {"Points_Winner","Points_Loser"} <= set(df.columns):
        pw = pd.to_numeric(df["Points_Winner"], errors="coerce")
        pl = pd.to_numeric(df["Points_Loser"], errors="coerce")
        df = df[pw.notna() & pl.notna()].copy()
    else:
        df = df.iloc[0:0].copy()  # no scores → nothing to append

    df["season"] = year
    df["Week"] = int(week)
    return df

def append_this_week(out_path="data/historical_results.csv"):
    """
    Detect the current season & week, scrape *this week's* completed games,
    and append them to out_path (de-duplicated).
    """
    year = pd.Timestamp.now().year
    week = _espn_current_week(year)
    if not week:
        print("⚠️  Could not detect current week from ESPN; skipping current-week append.")
        return

    df_week = _scrape_week_from_pfr(year, week)
    if df_week.empty:
        print(f"ℹ️  No completed games found for {year} Week {week} yet.")
        return

    # Append with de-duplication on sensible keys
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        key_cols = [c for c in ["season","Week","Date","Winner/tie","Loser/tie"] if c in df_week.columns and c in existing.columns]
        if key_cols:
            merged = pd.concat([existing, df_week], ignore_index=True).drop_duplicates(subset=key_cols, keep="first")
        else:
            merged = pd.concat([existing, df_week], ignore_index=True).drop_duplicates()
        merged.to_csv(out_path, index=False)
        print(f"✅ Appended {len(merged)-len(existing)} rows for {year} Week {week} to {out_path}")
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_week.to_csv(out_path, index=False)
        print(f"🆕 Created {out_path} with {len(df_week)} rows from {year} Week {week}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Keep your original behavior…
    scrape_historical(years=5)
    # …and then just add THIS WEEK on top.
    append_this_week("data/historical_results.csv")


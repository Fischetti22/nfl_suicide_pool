import pandas as pd

def scrape_season(year):
    """Scrape a full NFL season from Pro-Football-Reference, keeping only completed games."""
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
    # Some PFR tables use PtsW/PtsL instead of Pts/Pts.1 — normalize those too
    if "PtsW" in df.columns and "Points_Winner" not in df.columns:
        df = df.rename(columns={"PtsW": "Points_Winner"})
    if "PtsL" in df.columns and "Points_Loser" not in df.columns:
        df = df.rename(columns={"PtsL": "Points_Loser"})

    # keep only rows where both scores are present (completed games)
    if {"Points_Winner", "Points_Loser"} <= set(df.columns):
        pw = pd.to_numeric(df["Points_Winner"], errors="coerce")
        pl = pd.to_numeric(df["Points_Loser"], errors="coerce")
        df = df[pw.notna() & pl.notna()].copy()
    else:
        # If Points_* are still absent, attempt a last-chance filter on any numeric-looking score cols
        for wcol, lcol in [("PtsW","PtsL"), ("Pts","Pts.1")]:
            if {wcol, lcol} <= set(df.columns):
                pw = pd.to_numeric(df[wcol], errors="coerce")
                pl = pd.to_numeric(df[lcol], errors="coerce")
                df = df[pw.notna() & pl.notna()].copy()
                # normalize names for downstream consumers
                df = df.rename(columns={wcol: "Points_Winner", lcol: "Points_Loser"})
                break
    df["season"] = year
    return df

def scrape_historical(years=5, out_path="data/historical_results.csv"):
    current_year = pd.Timestamp.now().year
    frames = []
    # include the current season as well
    for y in range(current_year - years, current_year + 1):
        print(f"Fetching {y} season...")
        frames.append(scrape_season(y))

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(out_path, index=False)
    print(f"✅ Saved historical results to {out_path}")
import requests

def _espn_current_week(year: int) -> int | None:
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"seasontype": 2, "dates": year}
        data = requests.get(url, params=params, timeout=15).json()
        wk = (data.get("week") or {}).get("number")
        if isinstance(wk, int) and 1 <= wk <= 23:
            return wk
    except Exception:
        pass
    return None

def _pfr_max_completed_week(year: int) -> int | None:
    try:
        url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
        df = pd.read_html(url)[0]
        df = df[df["Week"].apply(lambda x: str(x).isdigit())].copy()
        # Only consider rows with scores (completed), so max reflects finished weeks
        if {"Pts","Pts.1"} <= set(df.columns):
            pw = pd.to_numeric(df["Pts"], errors="coerce")
            pl = pd.to_numeric(df["Pts.1"], errors="coerce")
            df = df[pw.notna() & pl.notna()]
        return int(df["Week"].astype(int).max())
    except Exception:
        return None

def _scrape_week_from_pfr(year: int, week: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{year}/week_{week}.htm"
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame()

    df = next((t for t in tables if {"Winner/tie","Loser/tie"}.issubset(set(map(str, t.columns)))), None)
    if df is None:
        return pd.DataFrame()

    # normalize to your column names
    df = df.rename(columns={
        "Pts": "Points_Winner",
        "Pts.1": "Points_Loser",
        "YdsW": "Yards_Winner",
        "YdsL": "Yards_Loser",
        "TOW": "Turnovers_Winner",
        "TOL": "Turnovers_Loser",
    })
    # completed only
    if {"Points_Winner","Points_Loser"} <= set(df.columns):
        pw = pd.to_numeric(df["Points_Winner"], errors="coerce")
        pl = pd.to_numeric(df["Points_Loser"], errors="coerce")
        df = df[pw.notna() & pl.notna()].copy()
    else:
        return pd.DataFrame()

    df["season"] = year
    df["Week"] = int(week)
    return df


if __name__ == "__main__":
    scrape_historical(years=5)  # your original behavior

    # --- add *this week* (ESPN detect; PFR fallback) ---
    year_now = pd.Timestamp.now().year
    wk = _espn_current_week(year_now) or _pfr_max_completed_week(year_now)
    if wk:
        df_week = _scrape_week_from_pfr(year_now, wk)  # PFR data as requested
        if not df_week.empty:
            out_path = "data/historical_results.csv"
            # de-dup append
            import os
            os.makedirs("data", exist_ok=True)
            if os.path.exists(out_path):
                existing = pd.read_csv(out_path)
                key_cols = [c for c in ["season","Week","Date","Winner/tie","Loser/tie"] if c in df_week.columns and c in existing.columns]
                if key_cols:
                    merged = pd.concat([existing, df_week], ignore_index=True).drop_duplicates(subset=key_cols, keep="first")
                else:
                    merged = pd.concat([existing, df_week], ignore_index=True).drop_duplicates()
                merged.to_csv(out_path, index=False)
                print(f"✅ Appended {len(merged)-len(existing)} rows for {year_now} Week {wk} (PFR).")
            else:
                df_week.to_csv(out_path, index=False)
                print(f"🆕 Created {out_path} with {len(df_week)} rows for {year_now} Week {wk} (PFR).")
    else:
        print("⚠️ Could not determine current week via ESPN or PFR.")


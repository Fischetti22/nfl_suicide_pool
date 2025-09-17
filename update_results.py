import requests
import pandas as pd

def fetch_from_espn(year, week):
    """Fetch games for a given season/week from ESPN"""
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
    """Fetch schedule from Pro-Football-Reference (works even before games are played)"""
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
    df = pd.read_html(url)[0]
    df = df[df["Week"].apply(lambda x: str(x).isdigit())]
    week_games = df[df["Week"].astype(int) == week]

    games = []
    for _, row in week_games.iterrows():
        # Determine home/away from '@' column
        if "@" in str(row.get("Unnamed: 5", "")):
            home_team = row["Loser/tie"]
            away_team = row["Winner/tie"]
        else:
            home_team = row["Winner/tie"]
            away_team = row["Loser/tie"]

        home_score = row.get("Pts.1") if "Pts.1" in row else None
        away_score = row.get("Pts") if "Pts" in row else None

        games.append({
            "date": row["Date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score if pd.notna(home_score) else None,
            "away_score": away_score if pd.notna(away_score) else None,
            "status": "Scheduled" if pd.isna(home_score) else "Final",
            "source": "PFR"
        })
    return games

def current_year_week(prefer_upcoming: bool = True):
    now = pd.Timestamp.now()
    year = now.year
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        j = requests.get(url, params={"year": year, "seasontype": 2}, timeout=20).json()
        wk = (j.get("week") or {}).get("number")
        if not (isinstance(wk, int) and 1 <= wk <= 23):
            return year, 1

        if prefer_upcoming:
            cw = requests.get(url, params={"year": year, "week": wk, "seasontype": 2}, timeout=20).json()
            events = cw.get("events", [])
            def is_final(ev):
                try:
                    st = ev["competitions"][0]["status"]["type"]
                    return st.get("state") == "post" or "final" in st.get("description", "").lower()
                except Exception:
                    return False
            if events and all(is_final(ev) for ev in events):
                nxt = requests.get(url, params={"year": year, "week": wk + 1, "seasontype": 2}, timeout=20).json()
                if nxt.get("events"):
                    return year, wk + 1
        return year, wk
    except Exception:
        return year, 1


def fetch_current_results(year=None, week=None):
    if year is None or week is None:
        y, w = current_year_week(prefer_upcoming=True)
        year = y if year is None else year
        week = w if week is None else week

    games = fetch_from_espn(year, week)

    if not games:
        print(f"⚠️ ESPN has no data for {year} Week {week}, falling back to PFR...")
        games = fetch_from_pfr(year, week)

    if not games:
        print(f"❌ No games found for {year} Week {week}")
        return

    df = pd.DataFrame(games)
    df.to_csv("data/current_results.csv", index=False)
    print(f"✅ Saved {len(games)} games for {year} Week {week} (source: {games[0]['source']})")

if __name__ == "__main__":
    fetch_current_results()


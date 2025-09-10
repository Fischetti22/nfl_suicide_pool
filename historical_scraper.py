import pandas as pd

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
    df_all.to_csv(out_path, index=False)
    print(f"✅ Saved historical results to {out_path}")

if __name__ == "__main__":
    scrape_historical(years=5)


# app_basic_team.py
import streamlit as st
import pandas as pd
from time import sleep
from typing import Dict, Tuple

# --- nba_api imports
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import leaguedashteamstats

st.set_page_config(page_title="NBA Team Basics — Season / Last 5 / Last 15", layout="wide")

# -----------------------
# Helpers & cached fetch
# -----------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def get_team_index() -> Tuple[pd.DataFrame, Dict[int, str], Dict[int, str]]:
    """
    Returns:
      teams_df: DataFrame with id, full_name, abbreviation
      id_to_name: TEAM_ID -> full_name
      id_to_abbr: TEAM_ID -> abbreviation
    """
    data = static_teams.get_teams()  # list of dicts
    teams_df = pd.DataFrame(data)
    teams_df = teams_df[["id", "full_name", "abbreviation"]].rename(
        columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"}
    )
    id_to_name = dict(teams_df[["TEAM_ID", "TEAM_NAME"]].values)
    id_to_abbr = dict(teams_df[["TEAM_ID", "TEAM_ABBREVIATION"]].values)
    return teams_df, id_to_name, id_to_abbr

def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Make numeric if possible; keep TEAM columns as objects
    for c in df.columns:
        if c not in ("TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_league_dash(season: str, last_n_games: int = 0) -> pd.DataFrame:
    """
    Fetch Base/PerGame team stats for a season with optional last_n_games.
    """
    # Simple retry wrapper (stats API can time out occasionally)
    for attempt in range(2):
        try:
            resp = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Base",
                per_mode_detailed="PerGame",
                last_n_games=last_n_games,
                timeout=30,
            )
            df = resp.get_data_frames()[0]
            return _numeric(df.copy())
        except Exception as e:
            if attempt == 0:
                sleep(1.0)
            else:
                raise e

def merge_team_labels(df: pd.DataFrame, id_to_name: Dict[int, str], id_to_abbr: Dict[int, str]) -> pd.DataFrame:
    """
    Ensures TEAM_NAME/TEAM_ABBREVIATION exist by mapping from TEAM_ID if necessary.
    """
    if "TEAM_ID" not in df.columns:
        return df  # Unexpected, but avoid crashing—let caller handle
    if "TEAM_NAME" not in df.columns or df["TEAM_NAME"].isna().any():
        df["TEAM_NAME"] = df["TEAM_ID"].map(id_to_name)
    if "TEAM_ABBREVIATION" not in df.columns or df["TEAM_ABBREVIATION"].isna().any():
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(id_to_abbr)
    return df

def select_team_row(df: pd.DataFrame, team_id: int) -> pd.Series:
    row = df.loc[df["TEAM_ID"] == team_id]
    if row.empty:
        raise RuntimeError(f"Selected TEAM_ID {team_id} not found in fetched table.")
    return row.iloc[0]

def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the core per-game columns first; show others after.
    """
    preferred = [
        "TEAM_NAME","TEAM_ABBREVIATION","GP","W","L","W_PCT","MIN","PTS",
        "FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","TOV","STL","BLK","BLKA","PF","PFD","PLUS_MINUS"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]

# -----------------------
# UI
# -----------------------
st.title("NBA Team Basics (Per-Game): Season, Last 5, Last 15")

# Season picker
season_default = "2025-26"  # update as needed
season = st.text_input("Season (format e.g. 2025-26)", value=season_default)

teams_df, id_to_name, id_to_abbr = get_team_index()

# Team select (by name)
team_name = st.selectbox(
    "Select Team",
    options=sorted(teams_df["TEAM_NAME"].unique()),
    index=sorted(teams_df["TEAM_NAME"].unique()).index("Boston Celtics") if "Boston Celtics" in set(teams_df["TEAM_NAME"]) else 0
)
team_id = int(teams_df.loc[teams_df["TEAM_NAME"] == team_name, "TEAM_ID"].iloc[0])

st.caption(f"TEAM_ID: {team_id}")

# -----------------------
# Fetch data blocks
# -----------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Season Averages")
    try:
        season_df = fetch_league_dash(season, last_n_games=0)
        season_df = merge_team_labels(season_df, id_to_name, id_to_abbr)
        season_df = tidy_columns(season_df)
        st.dataframe(season_df.loc[season_df["TEAM_ID"] == team_id].reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load season averages: {e}")

with col2:
    st.subheader("Last 5 Games")
    try:
        last5_df = fetch_league_dash(season, last_n_games=5)
        last5_df = merge_team_labels(last5_df, id_to_name, id_to_abbr)
        last5_df = tidy_columns(last5_df)
        st.dataframe(last5_df.loc[last5_df["TEAM_ID"] == team_id].reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load last 5: {e}")

with col3:
    st.subheader("Last 15 Games")
    try:
        last15_df = fetch_league_dash(season, last_n_games=15)
        last15_df = merge_team_labels(last15_df, id_to_name, id_to_abbr)
        last15_df = tidy_columns(last15_df)
        st.dataframe(last15_df.loc[last15_df["TEAM_ID"] == team_id].reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load last 15: {e}")

st.divider()
st.caption(
    "Notes: Data from nba_api LeagueDashTeamStats — measure=Base, per_mode=PerGame. "
    "We only read tables and display the chosen team’s row. Caching (10–30 min) reduces rate limits. "
    "If an endpoint times out, re-run once."
)

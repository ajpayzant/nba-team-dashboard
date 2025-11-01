# app_team_roster.py — NBA Team Roster Dashboard (season / last5 / last15)
# - Keeps original retry/cache/helpers structure where possible
# - Uses LeagueDashPlayerStats (PerGame) for season, last 5, and last 15
# - Select a Team; see all players on that roster at once (sorted by MIN)

import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt  # kept to mirror original imports (not required for tables)
import re
from zoneinfo import ZoneInfo

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Team Roster Dashboard", layout="wide")
st.title("NBA Team Roster Dashboard")

# ----------------------- Config -----------------------
CACHE_HOURS = 12
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    last_err = None
    for i in range(retries + 1):
        try:
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(sleep * (i + 1))
    raise last_err

def _season_labels(start=2010, end=None):
    if end is None:
        end = datetime.datetime.utcnow().year
    def lab(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start-1, -1)]

SEASONS = _season_labels(2015, datetime.datetime.utcnow().year)

# ----------------------- Utils -----------------------
def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: "{:.2f}" for c in num_cols}

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def _build_static_maps():
    teams_df = pd.DataFrame(static_teams.get_teams())
    id_by_full = dict(zip(teams_df["full_name"].astype(str), teams_df["id"].astype(int)))
    abbr_by_id = dict(zip(teams_df["id"].astype(int), teams_df["abbreviation"].astype(str)))
    name_by_id = dict(zip(teams_df["id"].astype(int), teams_df["full_name"].astype(str)))
    return teams_df, id_by_full, abbr_by_id, name_by_id

TEAMS_DF, TEAMID_BY_FULL, ABBR_BY_ID, NAME_BY_ID = _build_static_maps()

def merge_team_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TEAM_NAME / TEAM_ABBREVIATION exist using static maps."""
    if "TEAM_ID" not in df.columns:
        return df
    if "TEAM_NAME" not in df.columns or df["TEAM_NAME"].isna().any():
        df["TEAM_NAME"] = df["TEAM_ID"].map(NAME_BY_ID)
    if "TEAM_ABBREVIATION" not in df.columns or df["TEAM_ABBREVIATION"].isna().any():
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(ABBR_BY_ID)
    return df

# ----------------------- Cached data -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def fetch_players_per_game(season: str, last_n_games: int = 0) -> pd.DataFrame:
    """LeagueDashPlayerStats PerGame (whole league), optionally last_n_games."""
    kwargs = dict(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        last_n_games=last_n_games,
    )
    frames = _retry_api(LeagueDashPlayerStats, kwargs)
    df = frames[0] if frames else pd.DataFrame()
    if df.empty:
        return df
    # Coerce numerics where sensible
    for c in df.columns:
        if c not in ("PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return merge_team_labels(df)

# ----------------------- Sidebar (Season, Team) -----------------------
with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0, key="season_sel")

    team_name = st.selectbox(
        "Team",
        options=sorted(TEAMS_DF["full_name"].tolist()),
        index=sorted(TEAMS_DF["full_name"].tolist()).index("Boston Celtics") if "Boston Celtics" in TEAMS_DF["full_name"].tolist() else 0,
        key="team_sel"
    )
    team_id = TEAMID_BY_FULL.get(team_name, None)

if team_id is None:
    st.error("Could not resolve TEAM_ID for the selected team.")
    st.stop()

st.caption(f"TEAM: **{team_name}** — TEAM_ID: {team_id}")

# ----------------------- Fetch season / last5 / last15 -----------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Season Per-Game")
    try:
        season_pg = fetch_players_per_game(season, last_n_games=0)
        team_season = season_pg.loc[season_pg["TEAM_ID"] == team_id].copy()
        # tidy/ordering: show key box-score first
        prefer = [
            "PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","STL","BLK","TOV","PF","+/-"
        ]
        if "+/-" not in team_season.columns and "PLUS_MINUS" in team_season.columns:
            team_season.rename(columns={"PLUS_MINUS":"+/-"}, inplace=True)
        cols = [c for c in prefer if c in team_season.columns] + [c for c in team_season.columns if c not in prefer]
        team_season = team_season[cols].sort_values("MIN", ascending=False).reset_index(drop=True)
        st.dataframe(team_season.style.format(numeric_format_map(team_season)), use_container_width=True, height=_auto_height(team_season))
    except Exception as e:
        st.error(f"Failed to load season per-game: {e}")

with col2:
    st.subheader("Last 5 (Per-Game)")
    try:
        last5_pg = fetch_players_per_game(season, last_n_games=5)
        team_last5 = last5_pg.loc[last5_pg["TEAM_ID"] == team_id].copy()
        prefer = [
            "PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN","PTS","REB","AST","FG3M","FG3A","FTM","FTA","TOV","OREB","DREB"
        ]
        if "+/-" not in team_last5.columns and "PLUS_MINUS" in team_last5.columns:
            team_last5.rename(columns={"PLUS_MINUS":"+/-"}, inplace=True)
        cols = [c for c in prefer if c in team_last5.columns] + [c for c in team_last5.columns if c not in prefer]
        team_last5 = team_last5[cols].sort_values("MIN", ascending=False).reset_index(drop=True)
        st.dataframe(team_last5.style.format(numeric_format_map(team_last5)), use_container_width=True, height=_auto_height(team_last5))
    except Exception as e:
        st.error(f"Failed to load last 5 per-game: {e}")

with col3:
    st.subheader("Last 15 (Per-Game)")
    try:
        last15_pg = fetch_players_per_game(season, last_n_games=15)
        team_last15 = last15_pg.loc[last15_pg["TEAM_ID"] == team_id].copy()
        prefer = [
            "PLAYER_NAME","TEAM_ABBREVIATION","GP","MIN","PTS","REB","AST","FG3M","FG3A","FTM","FTA","TOV","OREB","DREB"
        ]
        if "+/-" not in team_last15.columns and "PLUS_MINUS" in team_last15.columns:
            team_last15.rename(columns={"PLUS_MINUS":"+/-"}, inplace=True)
        cols = [c for c in prefer if c in team_last15.columns] + [c for c in team_last15.columns if c not in prefer]
        team_last15 = team_last15[cols].sort_values("MIN", ascending=False).reset_index(drop=True)
        st.dataframe(team_last15.style.format(numeric_format_map(team_last15)), use_container_width=True, height=_auto_height(team_last15))
    except Exception as e:
        st.error(f"Failed to load last 15 per-game: {e}")

st.divider()
st.caption(
    "Source: LeagueDashPlayerStats (PerGame, Regular Season). "
    "We filter to the selected TEAM_ID and present the full roster at once, sorted by MIN. "
    "No opponent lookups or player-by-player loops are used here."
)

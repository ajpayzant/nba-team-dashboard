# app_team_roster.py — NBA Team Roster Dashboard (stacked tables; trimmed columns)
# - Uses LeagueDashPlayerStats (PerGame) for Season, Last 5, Last 15
# - Shows roster tables stacked vertically (easier to read)
# - Exact columns & order per spec; computes FG2M/FG2A if needed
# - Renames TEAM_ABBREVIATION -> TM and PLUS_MINUS -> +/- 

import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import LeagueDashPlayerStats

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

# ----------------------- Static Team Maps -----------------------
def _build_static_maps():
    teams_df = pd.DataFrame(static_teams.get_teams())
    id_by_full = dict(zip(teams_df["full_name"].astype(str), teams_df["id"].astype(int)))
    abbr_by_id = dict(zip(teams_df["id"].astype(int), teams_df["abbreviation"].astype(str)))
    name_by_id = dict(zip(teams_df["id"].astype(int), teams_df["full_name"].astype(str)))
    return teams_df, id_by_full, abbr_by_id, name_by_id

TEAMS_DF, TEAMID_BY_FULL, ABBR_BY_ID, NAME_BY_ID = _build_static_maps()

def merge_team_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "TEAM_ID" not in df.columns:
        return df
    if "TEAM_NAME" not in df.columns or df["TEAM_NAME"].isna().any():
        df["TEAM_NAME"] = df["TEAM_ID"].map(NAME_BY_ID)
    if "TEAM_ABBREVIATION" not in df.columns or df["TEAM_ABBREVIATION"].isna().any():
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(ABBR_BY_ID)
    return df

# ----------------------- Data Fetch -----------------------
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
    for c in df.columns:
        if c not in ("PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return merge_team_labels(df)

# ----------------------- Helpers -----------------------
COL_ORDER = [
    "TM", "PLAYER_NAME", "AGE", "GP", "MIN", "PTS", "REB", "AST",
    "FG2M", "FG2A", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "STL", "BLK", "TOV", "PF", "+/-"
]

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def _shape_roster_table(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    Filter to team, compute FG2M/FG2A, rename columns, and select exact order.
    """
    if df.empty:
        return df
    t = df.loc[df["TEAM_ID"] == team_id].copy()

    # Compute FG2M / FG2A if not present
    # LeagueDashPlayerStats columns typically include FGM, FGA, FG3M, FG3A
    if "FGM" in t.columns and "FG3M" in t.columns:
        t["FG2M"] = (t["FGM"] - t["FG3M"]).astype(float)
    else:
        t["FG2M"] = np.nan

    if "FGA" in t.columns and "FG3A" in t.columns:
        t["FG2A"] = (t["FGA"] - t["FG3A"]).astype(float)
    else:
        t["FG2A"] = np.nan

    # Ensure AGE exists (LeagueDashPlayerStats typically provides AGE)
    if "AGE" not in t.columns:
        t["AGE"] = np.nan

    # Rename TEAM_ABBREVIATION -> TM; PLUS_MINUS -> +/-
    if "TEAM_ABBREVIATION" in t.columns:
        t.rename(columns={"TEAM_ABBREVIATION": "TM"}, inplace=True)
    else:
        t["TM"] = t["TEAM_ID"].map(ABBR_BY_ID)

    if "+/-" not in t.columns:
        if "PLUS_MINUS" in t.columns:
            t.rename(columns={"PLUS_MINUS": "+/-"}, inplace=True)
        else:
            t["+/-"] = np.nan

    # Some required columns may not exist on some seasons; create if missing
    needed = set(COL_ORDER)
    for c in needed:
        if c not in t.columns:
            t[c] = np.nan

    # Keep only requested columns, in order; sort by MIN desc
    out = t[COL_ORDER].sort_values("MIN", ascending=False).reset_index(drop=True)
    return out

def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    fmts = {}
    for c in num_cols:
        fmts[c] = "{:.1f}" if c in {"MIN","PTS","REB","AST","FG2M","FG2A","FG3M","FG3A","FTM","FTA","OREB","DREB","STL","BLK","TOV"} else "{:.0f}"
    return fmts

# ----------------------- Sidebar -----------------------
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

# ----------------------- Season / Last 5 / Last 15 (stacked) -----------------------
# Season
st.subheader("Season Per-Game (Roster)")
try:
    season_pg = fetch_players_per_game(season, last_n_games=0)
    team_season = _shape_roster_table(season_pg, team_id)
    st.dataframe(
        team_season.style.format(numeric_format_map(team_season)),
        use_container_width=True,
        height=_auto_height(team_season)
    )
except Exception as e:
    st.error(f"Failed to load season per-game: {e}")

# Last 5
st.subheader("Last 5 Per-Game (Roster)")
try:
    last5_pg = fetch_players_per_game(season, last_n_games=5)
    team_last5 = _shape_roster_table(last5_pg, team_id)
    st.dataframe(
        team_last5.style.format(numeric_format_map(team_last5)),
        use_container_width=True,
        height=_auto_height(team_last5)
    )
except Exception as e:
    st.error(f"Failed to load last 5 per-game: {e}")

# Last 15
st.subheader("Last 15 Per-Game (Roster)")
try:
    last15_pg = fetch_players_per_game(season, last_n_games=15)
    team_last15 = _shape_roster_table(last15_pg, team_id)
    st.dataframe(
        team_last15.style.format(numeric_format_map(team_last15)),
        use_container_width=True,
        height=_auto_height(team_last15)
    )
except Exception as e:
    st.error(f"Failed to load last 15 per-game: {e}")

st.divider()
st.caption(
    "Source: LeagueDashPlayerStats (PerGame, Regular Season). "
    "Tables are stacked vertically and limited to the requested columns and order. "
    "FG2M/FG2A are computed as (FGM−FG3M)/(FGA−FG3A) when not present."
)

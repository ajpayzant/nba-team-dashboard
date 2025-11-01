# app_team_roster.py — NBA Team Roster Dashboard + Team Season Summary (ranks)
# - Adds a top "Team Season Summary" with record + value and league rank (1/30 best)
# - Traditional (PerGame): PTS, FG%, FGA, 3P%, 3PA, FT%, FTM, STL, BLK, TOV, +/-
# - Advanced (PerGame): OFF_RATING, DEF_RATING, NET_RATING, PACE
# - Three stacked player roster tables (Season / Last 5 / Last 15), same trimmed columns as before

import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import LeagueDashPlayerStats, leaguedashteamstats

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

# ----------------------- Data Fetch: Players -----------------------
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

# ----------------------- Data Fetch: Teams (Traditional + Advanced) -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def fetch_team_trad_advanced(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_kwargs = dict(
        season=season,
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        per_mode_detailed="PerGame",
    )
    # Traditional (PerGame)
    trad = _retry_api(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(base_kwargs, measure_type_detailed_defense="Base"),
    )
    trad = trad[0] if trad else pd.DataFrame()
    trad = merge_team_labels(trad)

    # Advanced (PerGame)
    adv = _retry_api(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(base_kwargs, measure_type_detailed_defense="Advanced"),
    )
    adv = adv[0] if adv else pd.DataFrame()
    adv = merge_team_labels(adv)

    # Coerce numerics
    for df in (trad, adv):
        for c in df.columns:
            if c not in ("TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

    return trad, adv

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
    if df.empty:
        return df
    t = df.loc[df["TEAM_ID"] == team_id].copy()

    if "FGM" in t.columns and "FG3M" in t.columns:
        t["FG2M"] = (t["FGM"] - t["FG3M"]).astype(float)
    else:
        t["FG2M"] = np.nan

    if "FGA" in t.columns and "FG3A" in t.columns:
        t["FG2A"] = (t["FGA"] - t["FG3A"]).astype(float)
    else:
        t["FG2A"] = np.nan

    if "AGE" not in t.columns:
        t["AGE"] = np.nan

    if "TEAM_ABBREVIATION" in t.columns:
        t.rename(columns={"TEAM_ABBREVIATION": "TM"}, inplace=True)
    else:
        t["TM"] = t["TEAM_ID"].map(ABBR_BY_ID)

    if "+/-" not in t.columns:
        if "PLUS_MINUS" in t.columns:
            t.rename(columns={"PLUS_MINUS": "+/-"}, inplace=True)
        else:
            t["+/-"] = np.nan

    needed = set(COL_ORDER)
    for c in needed:
        if c not in t.columns:
            t[c] = np.nan

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

# ----------------------- Team Season Summary (values + ranks) -----------------------
st.header(team_name)

try:
    trad_df, adv_df = fetch_team_trad_advanced(season)
    if trad_df.empty or adv_df.empty:
        raise RuntimeError("Failed to load team Traditional and/or Advanced tables.")

    # Merge Traditional + Advanced by TEAM_ID
    merged = pd.merge(
        trad_df[
            ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","GP","W","L","W_PCT","PTS",
             "FG_PCT","FGA","FG3_PCT","FG3A","FT_PCT","FTM","STL","BLK","TOV","PLUS_MINUS"]
        ],
        adv_df[["TEAM_ID","OFF_RATING","DEF_RATING","NET_RATING","PACE"]],
        on="TEAM_ID",
        how="inner"
    )

    team_row = merged.loc[merged["TEAM_ID"] == team_id].iloc[0]

    # Build league ranks (1 is best). Specify ascending per metric.
    metrics = {
        "PTS": False,
        "OFF_RATING": False,
        "DEF_RATING": True,   # lower is better
        "NET_RATING": False,
        "PACE": False,
        "FG_PCT": False,
        "FGA": False,
        "FG3_PCT": False,
        "FG3A": False,
        "FT_PCT": False,
        "FTM": False,
        "STL": False,
        "BLK": False,
        "TOV": True,          # lower is better
        "PLUS_MINUS": False,
    }

    # Compute ranks on merged league table
    ranks = {}
    for m, asc in metrics.items():
        ranks[m] = merged[m].rank(ascending=asc, method="min")

    n_teams = len(merged.index)

    def fmt_val(x, pct=False):
        if pd.isna(x):
            return "—"
        return f"{x:.3f}" if pct else f"{x:.1f}"

    def fmt_rank(r):
        if pd.isna(r):
            return "—"
        return f"{int(r)}/{n_teams}"

    record = f"{int(team_row['W'])}-{int(team_row['L'])}"

    # Prepare a tidy summary table with Value + Rank
    summary_rows = [
        ("Record", record, "—"),
        ("PTS", fmt_val(team_row["PTS"]), fmt_rank(ranks["PTS"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("NET Rating", fmt_val(team_row["NET_RATING"]), fmt_rank(ranks["NET_RATING"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("OFF Rating", fmt_val(team_row["OFF_RATING"]), fmt_rank(ranks["OFF_RATING"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("DEF Rating", fmt_val(team_row["DEF_RATING"]), fmt_rank(ranks["DEF_RATING"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("PACE", fmt_val(team_row["PACE"]), fmt_rank(ranks["PACE"].loc[merged["TEAM_ID"]==team_id].iloc[0])),

        ("FG%", fmt_val(team_row["FG_PCT"], pct=True), fmt_rank(ranks["FG_PCT"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("FGA", fmt_val(team_row["FGA"]), fmt_rank(ranks["FGA"].loc[merged["TEAM_ID"]==team_id].iloc[0])),

        ("3P%", fmt_val(team_row["FG3_PCT"], pct=True), fmt_rank(ranks["FG3_PCT"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("3PA", fmt_val(team_row["FG3A"]), fmt_rank(ranks["FG3A"].loc[merged["TEAM_ID"]==team_id].iloc[0])),

        ("FT%", fmt_val(team_row["FT_PCT"], pct=True), fmt_rank(ranks["FT_PCT"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("FTM", fmt_val(team_row["FTM"]), fmt_rank(ranks["FTM"].loc[merged["TEAM_ID"]==team_id].iloc[0])),

        ("STL", fmt_val(team_row["STL"]), fmt_rank(ranks["STL"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("BLK", fmt_val(team_row["BLK"]), fmt_rank(ranks["BLK"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("TOV", fmt_val(team_row["TOV"]), fmt_rank(ranks["TOV"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
        ("+/-", fmt_val(team_row["PLUS_MINUS"]), fmt_rank(ranks["PLUS_MINUS"].loc[merged["TEAM_ID"]==team_id].iloc[0])),
    ]

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value", "League Rank (1=best)"])

    st.subheader("Team Season Summary")
    st.dataframe(summary_df, use_container_width=True, height=_auto_height(summary_df))

except Exception as e:
    st.error(f"Unable to load Team Season Summary: {e}")

st.caption(f"Season: **{season}** • Data source: LeagueDashTeamStats (PerGame, Regular Season)")

# ----------------------- Roster Tables (stacked) -----------------------
st.markdown("---")
st.caption(f"TEAM: **{team_name}** — TEAM_ID: {team_id}")

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
    "Notes: Team summary uses NBA.com Traditional & Advanced (PerGame). "
    "League ranks treat lower DEF_RATING and TOV as better (1=best). "
    "Roster tables use LeagueDashPlayerStats (PerGame, Regular Season). "
    "FG2M/FG2A computed as (FGM−FG3M)/(FGA−FG3A) when not provided."
)

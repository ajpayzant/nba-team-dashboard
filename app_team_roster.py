# app_team_dashboard.py — NBA Team Dashboard (simple, stable, no opponent pulls)
# - Data sources (nba_api):
#   * LeagueDashTeamStats (Traditional=PerGame, Advanced=PerGame)
#   * LeagueDashPlayerStats (PerGame) with last_n_games in {0, 5, 15}
# - UI:
#   * Sidebar: Season, Team selector
#   * Header tiles: record, ratings, pace, shooting %, volumes, STL/BLK/TOV/+/-
#   * Stacked roster tables: Season, Last 5, Last 15
# - Tables show EXACT columns/order requested.

import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import LeagueDashPlayerStats, leaguedashteamstats

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Team Dashboard", layout="wide")
st.title("NBA Team Dashboard")

# ----------------------- Config -----------------------
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
CACHE_HOURS = 6

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

def _season_labels(start=2015, end=None):
    if end is None:
        end = dt.datetime.utcnow().year
    def lab(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    # show newest first
    return [lab(y) for y in range(end, start-1, -1)]

SEASONS = _season_labels(2015, dt.datetime.utcnow().year)

# ----------------------- Data Fetchers (cached) -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_df():
    """Static teams → DataFrame with id, full_name, abbreviation."""
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = t["TEAM_ID"].astype(int)
    return t[["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION"]]

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def fetch_league_team_traditional(season: str) -> pd.DataFrame:
    """Traditional team stats (PerGame)."""
    frames = _retry_api(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable="00",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
        ),
    )
    df = frames[0] if frames else pd.DataFrame()
    # NBA-only filter by TEAM_ID prefix
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def fetch_league_team_advanced(season: str) -> pd.DataFrame:
    """Advanced team stats (PerGame)."""
    frames = _retry_api(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable="00",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        ),
    )
    df = frames[0] if frames else pd.DataFrame()
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def fetch_league_players_pg(season: str, last_n_games: int) -> pd.DataFrame:
    """LeagueDashPlayerStats PerGame (last_n_games in {0,5,15})."""
    frames = _retry_api(
        LeagueDashPlayerStats,
        dict(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable="00",
            per_mode_detailed="PerGame",
            last_n_games=last_n_games,   # 0=season
        ),
    )
    df = frames[0] if frames else pd.DataFrame()
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    return df.reset_index(drop=True)

# ----------------------- Helpers -----------------------
def _fmt(v, pct=False, d=1):
    if pd.isna(v):
        return "—"
    if pct:
        return f"{float(v)*100:.{d}f}%"
    return f"{float(v):.{d}f}"

def _rank_series(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan]*len(df))
    return df[col].rank(ascending=ascending, method="min")

def _add_fg2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute FG2M and FG2A from FGM/FGA minus FG3M/FG3A if not present."""
    out = df.copy()
    if "FG2M" not in out.columns:
        out["FG2M"] = pd.to_numeric(out.get("FGM", 0), errors="coerce") - pd.to_numeric(out.get("FG3M", 0), errors="coerce")
    if "FG2A" not in out.columns:
        out["FG2A"] = pd.to_numeric(out.get("FGA", 0), errors="coerce") - pd.to_numeric(out.get("FG3A", 0), errors="coerce")
    return out

def _select_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Map NBA.com column names to requested order
    colmap = {
        "TEAM_ABBREVIATION": "TEAM",
        "PLAYER_NAME": "PLAYER_NAME",
        "AGE": "AGE",
        "GP": "GP",
        "MIN": "MIN",
        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "FG2M": "FG2M",
        "FG2A": "FG2A",
        "FG3M": "FG3M",
        "FG3A": "FG3A",
        "FTM": "FTM",
        "FTA": "FTA",
        "OREB": "OREB",
        "DREB": "DREB",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
        "PF": "PF",
        "PLUS_MINUS": "PLUS_MINUS",
    }
    for c in colmap.keys():
        if c not in df.columns:
            df[c] = np.nan
    out = df[list(colmap.keys())].copy()
    out.columns = list(colmap.values())
    return out

def _auto_height(df: pd.DataFrame, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("Filters")
    season = st.selectbox("Season", SEASONS, index=0)
    teams_df = get_teams_df()
    team_name = st.selectbox("Team", sorted(teams_df["TEAM_NAME"].tolist()))
    team_row = teams_df[teams_df["TEAM_NAME"] == team_name].iloc[0]
    team_id = int(team_row["TEAM_ID"])
    team_abbr = team_row["TEAM_ABBREVIATION"]

# ----------------------- Load league data -----------------------
with st.spinner("Loading league team stats..."):
    trad = fetch_league_team_traditional(season)
    adv = fetch_league_team_advanced(season)

if trad.empty or adv.empty:
    st.error("Could not load team stats. Try refreshing or changing the season.")
    st.stop()

# Merge traditional + advanced
merged = pd.merge(
    trad[
        [
            "TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","GP","W","L","W_PCT",
            "MIN","PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
            "OREB","DREB","REB","AST","STL","BLK","TOV","PLUS_MINUS"
        ]
    ],
    adv[["TEAM_ID","OFF_RATING","DEF_RATING","NET_RATING","PACE"]],
    on="TEAM_ID",
    how="left"
)

# League ranks (1 = best)
ranks = pd.DataFrame({"TEAM_ID": merged["TEAM_ID"]})
ranks["PTS"]         = _rank_series(merged, "PTS", ascending=False)
ranks["OFF_RATING"]  = _rank_series(merged, "OFF_RATING", ascending=False)
ranks["DEF_RATING"]  = _rank_series(merged, "DEF_RATING", ascending=True)
ranks["NET_RATING"]  = _rank_series(merged, "NET_RATING", ascending=False)
ranks["PACE"]        = _rank_series(merged, "PACE", ascending=False)
ranks["FG_PCT"]      = _rank_series(merged, "FG_PCT", ascending=False)
ranks["FGA"]         = _rank_series(merged, "FGA", ascending=False)
ranks["FG3_PCT"]     = _rank_series(merged, "FG3_PCT", ascending=False)
ranks["FG3A"]        = _rank_series(merged, "FG3A", ascending=False)
ranks["FT_PCT"]      = _rank_series(merged, "FT_PCT", ascending=False)
ranks["FTM"]         = _rank_series(merged, "FTM", ascending=False)
ranks["STL"]         = _rank_series(merged, "STL", ascending=False)
ranks["BLK"]         = _rank_series(merged, "BLK", ascending=False)
ranks["TOV"]         = _rank_series(merged, "TOV", ascending=True)   # lower TOV is better
ranks["PLUS_MINUS"]  = _rank_series(merged, "PLUS_MINUS", ascending=False)

n_teams = len(merged)

# Selected team row
sel = merged[merged["TEAM_ID"] == team_id]
if sel.empty:
    st.error("Selected team not found in this season dataset.")
    st.stop()

tr = sel.iloc[0]
rr = ranks[ranks["TEAM_ID"] == team_id].iloc[0]
record = f"{int(tr['W'])}–{int(tr['L'])}"

# ----------------------- Header -----------------------
st.subheader(f"{tr['TEAM_NAME']} — {season}")

# Metric tiles (clean visual, rank as small “delta”)
def _metric(col, label, value, rank, pct=False, d=1):
    val = _fmt(value, pct=pct, d=d)
    delta = f"Rank {int(rank)}/{n_teams}" if pd.notna(rank) else None
    col.metric(label, val, delta=delta)

# First line: Record
c_rec, _, _, _, _ = st.columns(5)
c_rec.metric("Record", record)

# Row 1: Scoring / ratings / pace
c1, c2, c3, c4, c5 = st.columns(5)
_metric(c1, "PTS",        tr["PTS"],        rr["PTS"])
_metric(c2, "NET Rating", tr["NET_RATING"], rr["NET_RATING"])
_metric(c3, "OFF Rating", tr["OFF_RATING"], rr["OFF_RATING"])
_metric(c4, "DEF Rating", tr["DEF_RATING"], rr["DEF_RATING"])
_metric(c5, "PACE",       tr["PACE"],       rr["PACE"])

# Row 2: FG / 3P / FT
c6, c7, c8, c9, c10 = st.columns(5)
_metric(c6,  "FG%",  tr["FG_PCT"],  rr["FG_PCT"],  pct=True)
_metric(c7,  "FGA",  tr["FGA"],     rr["FGA"])
_metric(c8,  "3P%",  tr["FG3_PCT"], rr["FG3_PCT"], pct=True)
_metric(c9,  "3PA",  tr["FG3A"],    rr["FG3A"])
_metric(c10, "FT%",  tr["FT_PCT"],  rr["FT_PCT"],  pct=True)

# Row 3: Makes + defense/misc
c11, c12, c13, c14, c15 = st.columns(5)
_metric(c11, "FTM",       tr["FTM"],        rr["FTM"])
_metric(c12, "STL",       tr["STL"],        rr["STL"])
_metric(c13, "BLK",       tr["BLK"],        rr["BLK"])
_metric(c14, "TOV",       tr["TOV"],        rr["TOV"])
_metric(c15, "+/-",       tr["PLUS_MINUS"], rr["PLUS_MINUS"])

st.caption("Ranks are relative to all NBA teams (1 = best). Shooting % tiles display percentage; volume tiles show per-game counts.")

# ----------------------- Roster tables (stacked) -----------------------
with st.spinner("Loading roster per-game (season / last 5 / last 15)..."):
    season_pg = fetch_league_players_pg(season, last_n_games=0)
    last5_pg  = fetch_league_players_pg(season, last_n_games=5)
    last15_pg = fetch_league_players_pg(season, last_n_games=15)

def _prep_roster(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[df["TEAM_ID"] == team_id].copy()
    if out.empty:
        return out
    # Ensure numeric for downstream formatting
    num_like = ["AGE","GP","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","STL","BLK","TOV","PF","PLUS_MINUS"]
    for c in num_like:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Compute FG2M/FG2A & select columns
    out = _add_fg2(out)
    out = _select_roster_columns(out)
    # Ordering: show by MIN desc
    if "MIN" in out.columns:
        out = out.sort_values("MIN", ascending=False).reset_index(drop=True)
    return out

season_tbl = _prep_roster(season_pg, team_id)
last5_tbl  = _prep_roster(last5_pg, team_id)
last15_tbl = _prep_roster(last15_pg, team_id)

def _num_fmt_map(df: pd.DataFrame):
    # One decimal for most counting/playing time stats
    fmts = {}
    for c in df.columns:
        if c in ("TEAM","PLAYER_NAME"): 
            continue
        fmts[c] = "{:.1f}"
    return fmts

st.markdown("### Roster — Season Per-Game")
if season_tbl.empty:
    st.info("No season per-game data for this team.")
else:
    st.dataframe(
        season_tbl.style.format(_num_fmt_map(season_tbl)),
        use_container_width=True,
        height=_auto_height(season_tbl),
    )

st.markdown("### Roster — Last 5 Games (Per-Game)")
if last5_tbl.empty:
    st.info("No Last 5 per-game data for this team.")
else:
    st.dataframe(
        last5_tbl.style.format(_num_fmt_map(last5_tbl)),
        use_container_width=True,
        height=_auto_height(last5_tbl),
    )

st.markdown("### Roster — Last 15 Games (Per-Game)")
if last15_tbl.empty:
    st.info("No Last 15 per-game data for this team.")
else:
    st.dataframe(
        last15_tbl.style.format(_num_fmt_map(last15_tbl)),
        use_container_width=True,
        height=_auto_height(last15_tbl),
    )

# ----------------------- Footer -----------------------
st.caption(
    "Notes: Team stats from NBA.com LeagueDashTeamStats (Traditional & Advanced, Per-Game). "
    "Player roster per-game from LeagueDashPlayerStats with last_n_games filters (0/5/15). "
    "FG2M/FG2A are computed as (FGM−FG3M)/(FGA−FG3A). Tables are sorted by MIN."
)

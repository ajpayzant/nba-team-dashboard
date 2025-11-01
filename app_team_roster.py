# app_team_dashboard.py — NBA Team Dashboard (defensive columns version)
# Stable: avoids opponent endpoints; guards against missing team columns.

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
    return [lab(y) for y in range(end, start-1, -1)]

SEASONS = _season_labels(2015, dt.datetime.utcnow().year)

# ----------------------- Data Fetchers (cached) -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_teams_df():
    t = pd.DataFrame(static_teams.get_teams())
    t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
    t["TEAM_ID"] = t["TEAM_ID"].astype(int)
    return t[["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION"]]

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def fetch_league_team_traditional(season: str) -> pd.DataFrame:
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
    if df.empty:
        return df
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    # normalize dtypes
    for c in df.columns:
        if c not in ("TEAM_NAME","TEAM_ABBREVIATION"):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df.reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def fetch_league_team_advanced(season: str) -> pd.DataFrame:
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
    if df.empty:
        return df
    df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
    for c in df.columns:
        if c not in ("TEAM_NAME","TEAM_ABBREVIATION"):
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df.reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
def fetch_league_players_pg(season: str, last_n_games: int) -> pd.DataFrame:
    frames = _retry_api(
        LeagueDashPlayerStats,
        dict(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable="00",
            per_mode_detailed="PerGame",
            last_n_games=last_n_games,   # 0=season, 5, 15
        ),
    )
    df = frames[0] if frames else pd.DataFrame()
    if df.empty:
        return df
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
    out = df.copy()
    if "FG2M" not in out.columns:
        out["FG2M"] = pd.to_numeric(out.get("FGM", 0), errors="coerce") - pd.to_numeric(out.get("FG3M", 0), errors="coerce")
    if "FG2A" not in out.columns:
        out["FG2A"] = pd.to_numeric(out.get("FGA", 0), errors="coerce") - pd.to_numeric(out.get("FG3A", 0), errors="coerce")
    return out

def _select_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
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

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Guarantee columns exist (fill NaN if absent) to avoid KeyError on selection."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

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

# Columns we want from each table (guarded)
TRAD_WANTED = [
    "TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","GP","W","L","W_PCT",
    "MIN","PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
    "OREB","DREB","REB","AST","STL","BLK","TOV","PLUS_MINUS"
]
ADV_WANTED = ["TEAM_ID","OFF_RATING","DEF_RATING","NET_RATING","PACE"]

trad_g = _ensure_cols(trad, TRAD_WANTED)[TRAD_WANTED].copy()
adv_g  = _ensure_cols(adv,  ADV_WANTED)[ADV_WANTED].copy()

# Merge traditional + advanced
merged = pd.merge(trad_g, adv_g, on="TEAM_ID", how="left")

# League ranks (1 = best); compute only if column exists
def _safe_rank(col, ascending):
    return _rank_series(merged, col, ascending=ascending)

ranks = pd.DataFrame({"TEAM_ID": merged["TEAM_ID"]})
ranks["PTS"]         = _safe_rank("PTS", ascending=False)
ranks["OFF_RATING"]  = _safe_rank("OFF_RATING", ascending=False)
ranks["DEF_RATING"]  = _safe_rank("DEF_RATING", ascending=True)
ranks["NET_RATING"]  = _safe_rank("NET_RATING", ascending=False)
ranks["PACE"]        = _safe_rank("PACE", ascending=False)
ranks["FG_PCT"]      = _safe_rank("FG_PCT", ascending=False)
ranks["FGA"]         = _safe_rank("FGA", ascending=False)
ranks["FG3_PCT"]     = _safe_rank("FG3_PCT", ascending=False)
ranks["FG3A"]        = _safe_rank("FG3A", ascending=False)
ranks["FT_PCT"]      = _safe_rank("FT_PCT", ascending=False)
ranks["FTM"]         = _safe_rank("FTM", ascending=False)
ranks["STL"]         = _safe_rank("STL", ascending=False)
ranks["BLK"]         = _safe_rank("BLK", ascending=False)
ranks["TOV"]         = _safe_rank("TOV", ascending=True)   # lower is better
ranks["PLUS_MINUS"]  = _safe_rank("PLUS_MINUS", ascending=False)

n_teams = len(merged)

# Selected team row
sel = merged[merged["TEAM_ID"] == team_id]
if sel.empty:
    st.error("Selected team not found in this season dataset.")
    st.stop()

tr = sel.iloc[0]
rr = ranks[ranks["TEAM_ID"] == team_id].iloc[0]
record = (
    f"{int(tr['W'])}–{int(tr['L'])}"
    if pd.notna(tr.get("W")) and pd.notna(tr.get("L"))
    else "—"
)

# ----------------------- Header -----------------------
st.subheader(f"{tr['TEAM_NAME']} — {season}")

def _metric(col, label, value, rank, pct=False, d=1):
    val = _fmt(value, pct=pct, d=d)
    delta = f"Rank {int(rank)}/{n_teams}" if pd.notna(rank) else None
    col.metric(label, val, delta=delta)

# First line: Record
c_rec, _, _, _, _ = st.columns(5)
c_rec.metric("Record", record)

# Row 1: Scoring / ratings / pace
c1, c2, c3, c4, c5 = st.columns(5)
_metric(c1, "PTS",        tr.get("PTS"),        rr.get("PTS"))
_metric(c2, "NET Rating", tr.get("NET_RATING"), rr.get("NET_RATING"))
_metric(c3, "OFF Rating", tr.get("OFF_RATING"), rr.get("OFF_RATING"))
_metric(c4, "DEF Rating", tr.get("DEF_RATING"), rr.get("DEF_RATING"))
_metric(c5, "PACE",       tr.get("PACE"),       rr.get("PACE"))

# Row 2: FG / 3P / FT
c6, c7, c8, c9, c10 = st.columns(5)
_metric(c6,  "FG%",  tr.get("FG_PCT"),  rr.get("FG_PCT"),  pct=True)
_metric(c7,  "FGA",  tr.get("FGA"),     rr.get("FGA"))
_metric(c8,  "3P%",  tr.get("FG3_PCT"), rr.get("FG3_PCT"), pct=True)
_metric(c9,  "3PA",  tr.get("FG3A"),    rr.get("FG3A"))
_metric(c10, "FT%",  tr.get("FT_PCT"),  rr.get("FT_PCT"),  pct=True)

# Row 3: Makes + defense/misc
c11, c12, c13, c14, c15 = st.columns(5)
_metric(c11, "FTM",       tr.get("FTM"),        rr.get("FTM"))
_metric(c12, "STL",       tr.get("STL"),        rr.get("STL"))
_metric(c13, "BLK",       tr.get("BLK"),        rr.get("BLK"))
_metric(c14, "TOV",       tr.get("TOV"),        rr.get("TOV"))
_metric(c15, "+/-",       tr.get("PLUS_MINUS"), rr.get("PLUS_MINUS"))

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
    num_like = ["AGE","GP","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","STL","BLK","TOV","PF","PLUS_MINUS"]
    for c in num_like:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = _add_fg2(out)
    out = _select_roster_columns(out)
    if "MIN" in out.columns:
        out = out.sort_values("MIN", ascending=False).reset_index(drop=True)
    return out

def _num_fmt_map(df: pd.DataFrame):
    fmts = {}
    for c in df.columns:
        if c in ("TEAM","PLAYER_NAME"):
            continue
        fmts[c] = "{:.1f}"
    return fmts

season_tbl = _prep_roster(season_pg, team_id)
last5_tbl  = _prep_roster(last5_pg, team_id)
last15_tbl = _prep_roster(last15_pg, team_id)

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

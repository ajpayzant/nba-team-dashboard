import time
import numpy as np
import pandas as pd
import streamlit as st

# --- Harden nba_api HTTP so stats.nba.com behaves ---
from nba_api.stats.library.http import NBAStatsHTTP
NBAStatsHTTP.timeout = 30
NBAStatsHTTP.headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.static import teams as nba_static_teams

st.set_page_config(page_title="NBA Opponent Team Stats (Sanity Check)", layout="wide")
st.title("NBA Opponent Team Stats — Per Game (Sanity Check)")
st.caption("Goal: fetch and display the exact opponent-allowed per-game table that we’ll use in the main app.")

# ---------- Helpers ----------
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2018, 2026)]
DEFAULT_SEASON = SEASONS[-1]

def to_num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("TEAM_NAME", "TEAM_ABBREVIATION"):
            continue
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

def derive_2p(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    fgm, fga = f"{prefix}FGM" or "FGM", f"{prefix}FGA" or "FGA"
    fg3m, fg3a = f"{prefix}FG3M" or "FG3M", f"{prefix}FG3A" or "FG3A"
    p2m = f"{prefix}_2PM" if prefix else "_2PM"
    p2a = f"{prefix}_2PA" if prefix else "_2PA"
    if fgm in df.columns and fg3m in df.columns:
        df[p2m] = df[fgm] - df[fg3m]
    if fga in df.columns and fg3a in df.columns:
        df[p2a] = df[fga] - df[fg3a]
    return df

def ensure_team_keys(df: pd.DataFrame) -> pd.DataFrame:
    # Sometimes names are present but IDs aren’t; backfill from static map
    if "TEAM_ID" not in df.columns:
        static = pd.DataFrame(nba_static_teams.get_teams())
        df = df.merge(static[["id","full_name","abbreviation"]],
                      left_on="TEAM_NAME", right_on="full_name", how="left")
        df = df.rename(columns={"id":"TEAM_ID","abbreviation":"TEAM_ABBREVIATION"})
        df.drop(columns=["full_name"], inplace=True, errors="ignore")
    return df

def fetch_opp_per_game(season: str) -> tuple[pd.DataFrame, str]:
    """
    Primary: Opponent / PerGame
    Fallback: Opponent / Totals -> per-game
    Returns (df, source_label)
    """
    # Try Opponent / PerGame
    try:
        df = LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Opponent",
        ).get_data_frames()[0]
        df = ensure_team_keys(to_num(df))
        df = derive_2p(df)
        need = ["TEAM_ID","TEAM_NAME","GP","PTS","FGM","FGA","FG3M","FG3A","FTM","FTA","REB","AST","TOV"]
        if all(c in df.columns for c in need):
            return df, "Opponent/PerGame"
        st.warning("Opponent/PerGame returned but missing core columns — will try fallback.")
    except Exception as e:
        st.warning(f"Opponent/PerGame failed: {e}")

    # Fallback Opponent / Totals → PerGame
    try:
        tot = LeagueDashTeamStats(
            season=season,
            per_mode_detailed="Totals",
            measure_type_detailed_defense="Opponent",
        ).get_data_frames()[0]
        tot = ensure_team_keys(to_num(tot))
        need = ["TEAM_ID","TEAM_NAME","GP","PTS","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB","AST","TOV"]
        if all(c in tot.columns for c in need):
            pg = tot.copy()
            pg["GP"] = pg["GP"].replace({0: np.nan})
            for c in ["PTS","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB","AST","TOV"]:
                pg[c] = pg[c] / pg["GP"]
            pg = derive_2p(pg)
            return pg, "Opponent/Totals→PerGame"
        else:
            st.error("Opponent/Totals present but missing required columns.")
    except Exception as e:
        st.error(f"Opponent/Totals fetch failed: {e}")

    # Total failure
    return pd.DataFrame(), "NONE"

# ---------- UI controls ----------
c1, c2 = st.columns([1,1])
with c1:
    season = st.selectbox("Season", SEASONS, index=SEASONS.index(DEFAULT_SEASON))
with c2:
    sort_col = st.selectbox("Sort by (allowed stat)", 
                            ["PTS","FGA","FGM","FG3A","FG3M","FTA","FTM","REB","OREB","DREB","AST","TOV","_2PA","_2PM"],
                            index=0)

st.divider()

# ---------- Fetch + show ----------
t0 = time.time()
df, source = fetch_opp_per_game(season)
elapsed = time.time() - t0

st.subheader("Status")
if source == "NONE" or df.empty:
    st.error("🚫 Could not retrieve opponent team table from NBA stats. Check network / headers / rate limits.")
    st.stop()
else:
    st.success(f"✅ Retrieved opponent team table from: **{source}**  |  {len(df)} rows  |  {elapsed:.2f}s")

# Core subset columns to display cleanly
show_cols = ["TEAM_ID","TEAM_NAME","GP","PTS","FGM","FGA","FG3M","FG3A","_2PM","_2PA","FTM","FTA","OREB","DREB","REB","AST","TOV"]
present = [c for c in show_cols if c in df.columns]
missing = [c for c in show_cols if c not in df.columns]
if missing:
    st.warning(f"Columns missing from response (FYI): {missing}")

# Sort and show
if sort_col not in df.columns:
    st.info(f"Sort column '{sort_col}' not found in data; showing unsorted.")
    view = df[present].copy()
else:
    view = df[present].sort_values(sort_col, ascending=False).reset_index(drop=True)

st.markdown("### Opponent Team Stats — Per Game (Volumes Only)")
st.dataframe(view, use_container_width=True, hide_index=True)

# Download as CSV to verify schema
csv = view.to_csv(index=False)
st.download_button("Download CSV (opponent_per_game.csv)", data=csv, file_name="opponent_per_game.csv", mime="text/csv")

# Raw frame (debug)
with st.expander("Raw response (debug)"):
    st.dataframe(df, use_container_width=True)

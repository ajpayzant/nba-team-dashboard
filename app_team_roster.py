# --- Pretty metric tiles instead of a table ---

st.subheader("Team Season Summary")

def _rank_for(metric: str):
    try:
        return int(ranks[metric].loc[merged["TEAM_ID"] == team_id].iloc[0])
    except Exception:
        return None

def _val_fmt(x, pct=False, decimals=1):
    if pd.isna(x): 
        return "—"
    if pct:
        return f"{x*100:.{decimals}f}%"
    return f"{x:.{decimals}f}"

def _metric(col, label, value, rank, pct=False, decimals=1):
    val_str = _val_fmt(value, pct=pct, decimals=decimals)
    delta_str = f"Rank {rank}/{n_teams}" if isinstance(rank, int) else None
    col.metric(label=label, value=val_str, delta=delta_str)

# Top line: Record as its own tile
c_rec, _, _, _, _ = st.columns(5)
c_rec.metric("Record", record)

# Row 1: Scoring/pace & ratings
c1, c2, c3, c4, c5 = st.columns(5)
_metric(c1, "PTS",        team_row["PTS"],        _rank_for("PTS"))
_metric(c2, "NET Rating", team_row["NET_RATING"], _rank_for("NET_RATING"))
_metric(c3, "OFF Rating", team_row["OFF_RATING"], _rank_for("OFF_RATING"))
_metric(c4, "DEF Rating", team_row["DEF_RATING"], _rank_for("DEF_RATING"))
_metric(c5, "PACE",       team_row["PACE"],       _rank_for("PACE"))

# Row 2: Shooting volume/efficiency
c6, c7, c8, c9, c10 = st.columns(5)
_metric(c6,  "FG%",  team_row["FG_PCT"],  _rank_for("FG_PCT"),  pct=True,  decimals=1)
_metric(c7,  "FGA",  team_row["FGA"],     _rank_for("FGA"))
_metric(c8,  "3P%",  team_row["FG3_PCT"], _rank_for("FG3_PCT"), pct=True,  decimals=1)
_metric(c9,  "3PA",  team_row["FG3A"],    _rank_for("FG3A"))
_metric(c10, "FT%",  team_row["FT_PCT"],  _rank_for("FT_PCT"),  pct=True,  decimals=1)

# Row 3: Makes + defense/misc
c11, c12, c13, c14, c15 = st.columns(5)
_metric(c11, "FTM",       team_row["FTM"],        _rank_for("FTM"))
_metric(c12, "STL",       team_row["STL"],        _rank_for("STL"))
_metric(c13, "BLK",       team_row["BLK"],        _rank_for("BLK"))
_metric(c14, "TOV",       team_row["TOV"],        _rank_for("TOV"))
_metric(c15, "+/-",       team_row["PLUS_MINUS"], _rank_for("PLUS_MINUS"))

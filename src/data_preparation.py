"""
Phase 1: Data Collection & Preparation
FIFA World Cup 2026 Match Outcome Predictor
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: LOAD RAW DATA
# ─────────────────────────────────────────

def load_match_results() -> pd.DataFrame:
    path = RAW_DIR / "results.csv"

    df = pd.read_csv(path)

    # Create a *new* DataFrame column 'date' so no in-place replacement
    dates = pd.to_datetime(df['date'], errors='coerce')

    # Use pd.concat to add date column freshly
    df = pd.concat([df.drop(columns=['date']), dates.rename('date')], axis=1)

    # Reorder columns to place date first (optional)
    cols = ['date'] + [c for c in df.columns if c != 'date']
    df = df[cols]

    print(f"✅ Loaded match results:  {len(df):>7,} rows  | "
          f"date range {df['date'].min().date()} -> {df['date'].max().date()}")

    return df




def load_fifa_rankings() -> pd.DataFrame:
    """
    Load FIFA world rankings from fifa_rankings.csv.

    Schema quirks fixed here:
      - 'date' is an int year (e.g. 2024), 'semester' is 1 or 2
        -> we convert these into an approximate calendar date:
           semester 1 -> Jan 1,  semester 2 -> Jul 1
      - Points column is named 'total.points' (dot, not underscore)
      - Team names are in 'team' (not 'country_full')
    """
    path = RAW_DIR / "fifa_rankings.csv"
    df = pd.read_csv(path)

    # Build a proper date from year + semester
    df["rank_date"] = pd.to_datetime(
        df["date"].astype(str) + "-" +
        df["semester"].map({1: "01-01", 2: "07-01"})
    )

    # Rename for consistency with the rest of the pipeline
    df = df.rename(columns={
        "team":         "country_full",
        "total.points": "total_points",
    })

    print(f"✅ Loaded FIFA rankings:   {len(df):>7,} rows | "
          f"date range {df['rank_date'].min().date()} -> {df['rank_date'].max().date()}")
    return df

# ─────────────────────────────────────────
# STEP 2: CLEAN & FILTER
# ─────────────────────────────────────────

def clean_match_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['date', 'home_score', 'away_score']).copy()
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df = df[df['date'] >= '1992-01-01'].copy()
    df['goal_diff'] = df['home_score'] - df['away_score']
    df['result'] = np.select([df['goal_diff'] > 0, df['goal_diff'] < 0], ['H', 'A'], default='D')
    df['is_neutral'] = df['neutral'].astype(int)
    df['is_world_cup'] = df['tournament'].str.contains('FIFA World Cup', case=False, na=False).astype(int)
    df = df.reset_index(drop=True)
    print(f"✅ Cleaned results: {len(df):>7} rows (post-1992 filter applied)")
    return df




# ─────────────────────────────────────────
# STEP 3: FIFA RANKING LOOKUP (vectorised)
# ─────────────────────────────────────────

def build_ranking_lookup(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-sort rankings so we can use merge_asof for fast,
    leak-free lookups (most recent rank BEFORE match date).
    """
    return (
        rankings[["rank_date", "country_full", "rank", "total_points"]]
        .sort_values(["country_full", "rank_date"])
        .reset_index(drop=True)
    )

def attach_rankings(matches: pd.DataFrame,
                    ranking_lookup: pd.DataFrame,
                    side: str) -> pd.DataFrame:
    """
    For every match row, find the most recent FIFA rank/points
    for one side ('home' or 'away') using vectorised merge_asof.
    """
    team_col = f"{side}_team"
    rank_col  = f"{side}_fifa_rank"
    pts_col   = f"{side}_fifa_points"

    match_keys = (
        matches[["date", team_col]]
        .copy()
        .rename(columns={team_col: "country_full"})
        .reset_index()
        .sort_values("date")
    )

    results_list = []
    for team, group in match_keys.groupby("country_full", sort=False):
        team_ranks = (
            ranking_lookup[ranking_lookup["country_full"] == team]
            .sort_values("rank_date")
        )

        if team_ranks.empty:
            group[rank_col] = np.nan
            group[pts_col]  = np.nan
            results_list.append(group[["index", rank_col, pts_col]])
            continue

        merged = pd.merge_asof(
            group.sort_values("date"),
            team_ranks[["rank_date", "rank", "total_points"]].rename(
                columns={
                    "rank_date":    "date",
                    "rank":         rank_col,
                    "total_points": pts_col,
                }
            ),
            on="date",
            direction="backward",
        )
        results_list.append(merged[["index", rank_col, pts_col]])

    rank_df = pd.concat(results_list).set_index("index").sort_index()
    return matches.join(rank_df)

# ─────────────────────────────────────────
# STEP 4: ROLLING TEAM STATISTICS
# ─────────────────────────────────────────

def compute_team_rolling_stats(df: pd.DataFrame,
                                window: int = 10) -> pd.DataFrame:
    """
    For each team, compute rolling statistics over their last `window` matches.
    """
    # Reshape: one row per team per match
    home = df[["date", "home_team", "away_team",
               "home_score", "away_score", "result"]].copy()
    home = home.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "gf",  "away_score": "ga",
    })
    home["won"]  = (home["result"] == "H").astype(int)
    home["drew"] = (home["result"] == "D").astype(int)

    away = df[["date", "home_team", "away_team",
               "home_score", "away_score", "result"]].copy()
    away = away.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "gf", "home_score": "ga",
    })
    away["won"]  = (away["result"] == "A").astype(int)
    away["drew"] = (away["result"] == "D").astype(int)

    all_matches = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "date"])
    )

    all_matches["points"]      = all_matches["won"] * 3 + all_matches["drew"]
    all_matches["clean_sheet"] = (all_matches["ga"] == 0).astype(int)

    roll_cols = ["gf", "ga", "won", "drew", "points", "clean_sheet"]
    for col in roll_cols:
        all_matches[f"roll_{col}"] = (
            all_matches
            .groupby("team")[col]
            .transform(
                lambda x: x.shift(1)
                            .rolling(window, min_periods=3)
                            .mean()
            )
        )

    team_stats = (
        all_matches[["date", "team",
                     "roll_gf", "roll_ga", "roll_won", "roll_drew",
                     "roll_points", "roll_clean_sheet"]]
        .rename(columns={
            "roll_gf":          "avg_goals_scored",
            "roll_ga":          "avg_goals_conceded",
            "roll_won":         "win_rate",
            "roll_drew":        "draw_rate",
            "roll_points":      "avg_points",
            "roll_clean_sheet": "clean_sheet_rate",
        })
    )

    print(f"✅ Rolling team stats:     {len(team_stats):>7,} rows  (window={window})")
    return team_stats

# ─────────────────────────────────────────
# STEP 5: HEAD-TO-HEAD FEATURES
# ─────────────────────────────────────────

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute cumulative head-to-head record.
    """
    df = df.reset_index(drop=True)
    records = []

    for idx, row in df.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["date"]

        prior = df[
            (df["date"] < date) &
            (
                ((df["home_team"] == home) & (df["away_team"] == away)) |
                ((df["home_team"] == away) & (df["away_team"] == home))
            )
        ]

        total = len(prior)
        if total == 0:
            records.append({
                "h2h_home_win_rate": 0.5,
                "h2h_away_win_rate": 0.5,
                "h2h_draw_rate":     0.0,
                "h2h_total":         0,
            })
            continue

        home_wins = len(prior[
            ((prior["home_team"] == home) & (prior["result"] == "H")) |
            ((prior["away_team"] == home) & (prior["result"] == "A"))
        ])
        away_wins = len(prior[
            ((prior["home_team"] == away) & (prior["result"] == "H")) |
            ((prior["away_team"] == away) & (prior["result"] == "A"))
        ])
        draws = total - home_wins - away_wins

        records.append({
            "h2h_home_win_rate": home_wins / total,
            "h2h_away_win_rate": away_wins / total,
            "h2h_draw_rate":     draws     / total,
            "h2h_total":         total,
        })

    print(f"✅ H2H features computed:  {len(records):>7,} rows")
    return pd.DataFrame(records)

# ─────────────────────────────────────────
# STEP 6: ASSEMBLE FEATURE DATASET
# ─────────────────────────────────────────

FEATURE_COLS = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate",         "home_draw_rate",
    "home_avg_points",       "home_clean_sheet_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate",         "away_draw_rate",
    "away_avg_points",       "away_clean_sheet_rate",
    "h2h_home_win_rate", "h2h_away_win_rate",
    "h2h_draw_rate",     "h2h_total",
    "home_fifa_rank",   "home_fifa_points",
    "away_fifa_rank",   "away_fifa_points",
    "rank_diff",        "points_diff",
    "is_world_cup", "is_neutral",
]

TARGET_COLS = ["result", "result_encoded", "home_score", "away_score"]

def build_feature_dataset(matches: pd.DataFrame,
                           team_stats: pd.DataFrame,
                           rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Join rolling stats, FIFA rankings, and H2H features.
    """
    # Rolling stats -- home side
    home_stats = (
        team_stats
        .rename(columns={c: f"home_{c}"
                         for c in team_stats.columns
                         if c not in ("date", "team")})
        .rename(columns={"team": "home_team"})
    )
    df = matches.merge(home_stats, on=["date", "home_team"], how="left")

    # Rolling stats -- away side
    away_stats = (
        team_stats
        .rename(columns={c: f"away_{c}"
                         for c in team_stats.columns
                         if c not in ("date", "team")})
        .rename(columns={"team": "away_team"})
    )
    df = df.merge(away_stats, on=["date", "away_team"], how="left")

    # FIFA rankings
    ranking_lookup = build_ranking_lookup(rankings)
    df = df.sort_values("date").reset_index(drop=True)
    df = attach_rankings(df, ranking_lookup, side="home")
    df = attach_rankings(df, ranking_lookup, side="away")

    df["rank_diff"]   = df["away_fifa_rank"]   - df["home_fifa_rank"]
    df["points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]

    # Head-to-head
    print("⏳ Computing H2H features (may take ~30 s on 30 k rows)...")
    h2h = compute_h2h_features(df)
    df  = pd.concat([df.reset_index(drop=True),
                     h2h.reset_index(drop=True)], axis=1)

    # Encode outcome
    df["result_encoded"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    print(f"✅ Feature dataset built:  {len(df):>7,} rows x {df.shape} columns")
    return df

def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select feature + target columns; drop rows with NaN features.
    """
    keep      = ["date", "home_team", "away_team"] + FEATURE_COLS + TARGET_COLS
    available = [c for c in keep if c in df.columns]
    final     = df[available].dropna(subset=FEATURE_COLS, how="any")

    print(f"✅ Final dataset:          {len(final):>7,} rows  "
          f"(dropped {len(df) - len(final):,} rows with NaN features)")
    return final

# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    print("\n🚀 Starting Data Preparation Pipeline\n" + "=" * 50)

    matches  = load_match_results()
    rankings = load_fifa_rankings()

    matches    = clean_match_results(matches)
    team_stats = compute_team_rolling_stats(matches, window=10)
    features   = build_feature_dataset(matches, team_stats, rankings)
    final      = prepare_final_dataset(features)

    out_path = PROCESSED_DIR / "features.csv"
    final.to_csv(out_path, index=False)
    print(f"\n💾 Saved -> {out_path}")

    print("\n📊 Feature Summary:")
    print(final[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(3)
          .to_string())

    print("\n🎯 Target Distribution:")
    print(final["result"].value_counts().to_string())

    return final

if __name__ == "__main__":
    df = run_pipeline()

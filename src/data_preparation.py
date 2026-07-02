"""
Phase 1: Data Collection & Preparation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_match_results() -> pd.DataFrame:
    path = RAW_DIR / "results.csv"

    df = pd.read_csv(path)

    dates = pd.to_datetime(df['date'], errors='coerce')

    df = pd.concat([df.drop(columns=['date']), dates.rename('date')], axis=1)

    cols = ['date'] + [c for c in df.columns if c != 'date']
    df = df[cols]

    print(f"= Loaded match results:  {len(df):>7,} rows  | "
          f"date range {df['date'].min().date()} -> {df['date'].max().date()}")

    return df




# results.csv and fifa_rankings.csv use different naming conventions for the same countries. 
# most of those are either non-FIFA entities (Guernsey, Alderney, Shetland)
# or defunct teams (German DR / East Germany, Vietnam Republic / South Vietnam) with no modern ranking. 
# of the 48 confirmed 2026 WC teams, 6 of them: Cape Verde, DR Congo, Iran, Ivory Coast, South Korea, United States.
# matches results.csv spelling to fifa_rankings.csv spelling

RESULTS_TO_RANKINGS_ALIASES = {
    "United States":          "USA",
    "South Korea":            "Korea Republic",
    "North Korea":            "Korea DPR",
    "Ivory Coast":            "Côte d'Ivoire",
    "Iran":                   "IR Iran",
    "Cape Verde":             "Cabo Verde",
    "DR Congo":               "Congo DR",
    "China":                  "China PR",
}


def load_fifa_rankings() -> pd.DataFrame:
    """
    Load world rankings from fifa_rankings.csv.
    """
    path = RAW_DIR / "fifa_rankings.csv"
    df = pd.read_csv(path)

    df["rank_date"] = pd.to_datetime(
        df["date"].astype(str) + "-" +
        df["semester"].map({1: "01-01", 2: "07-01"})
    )

    df = df.rename(columns={
        "team":         "country_full",
        "total.points": "total_points",
    })

    print(f"= Loaded FIFA rankings:   {len(df):>7,} rows | "
          f"date range {df['rank_date'].min().date()} -> {df['rank_date'].max().date()}")
    return df


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
    print(f"= Cleaned results: {len(df):>7} rows (post-1992 filter applied)")
    return df


def build_ranking_lookup(rankings: pd.DataFrame) -> pd.DataFrame:
    return (
        rankings[["rank_date", "country_full", "rank", "total_points"]]
        .sort_values(["country_full", "rank_date"])
        .reset_index(drop=True)
    )

def attach_rankings(matches: pd.DataFrame,
                    ranking_lookup: pd.DataFrame,
                    side: str) -> pd.DataFrame:
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

    match_keys["country_full"] = match_keys["country_full"].replace(
        RESULTS_TO_RANKINGS_ALIASES
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


def compute_team_rolling_stats(df: pd.DataFrame,
                                window: int = 10,
                                form_window: int = 5) -> pd.DataFrame:
    """
    Compute rolling statistics over their last 10 games for long run and last 5 games for current form
    Also computes win/loss streaks

    Current form and long-run form are both needed in cases where teams may have a solid 10-game average but are middle of a 4-game losing or winning streak
    Having two form windows averages this kind of skew away
    """

    home = df[["date", "home_team", "away_team",
               "home_score", "away_score", "result"]].copy()
    home = home.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "gf",  "away_score": "ga",
    })
    home["won"]  = (home["result"] == "H").astype(int)
    home["drew"] = (home["result"] == "D").astype(int)
    home["lost"] = (home["result"] == "A").astype(int)

    away = df[["date", "home_team", "away_team",
               "home_score", "away_score", "result"]].copy()
    away = away.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "gf", "home_score": "ga",
    })
    away["won"]  = (away["result"] == "A").astype(int)
    away["drew"] = (away["result"] == "D").astype(int)
    away["lost"] = (away["result"] == "H").astype(int)

    all_matches = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "date"])
    )

    all_matches["points"]      = all_matches["won"] * 3 + all_matches["drew"]
    all_matches["clean_sheet"] = (all_matches["ga"] == 0).astype(int)
    all_matches["goal_diff"]   = all_matches["gf"] - all_matches["ga"]


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

    all_matches["form_points"] = (
        all_matches.groupby("team")["points"]
        .transform(lambda x: x.shift(1).rolling(form_window, min_periods=2).sum())
    )
    all_matches["form_goal_diff"] = (
        all_matches.groupby("team")["goal_diff"]
        .transform(lambda x: x.shift(1).rolling(form_window, min_periods=2).sum())
    )
    all_matches["form_goals_scored"] = (
        all_matches.groupby("team")["gf"]
        .transform(lambda x: x.shift(1).rolling(form_window, min_periods=2).mean())
    )
    all_matches["form_goals_conceded"] = (
        all_matches.groupby("team")["ga"]
        .transform(lambda x: x.shift(1).rolling(form_window, min_periods=2).mean())
    )


    def _streak(sub: pd.DataFrame) -> pd.Series:
        prior_result = sub["result"].shift(1)
        outcome = pd.Series(
            np.select(
                [sub["won"].shift(1) == 1, sub["lost"].shift(1) == 1],
                ["W", "L"],
                default="D",
            ),
            index=sub.index,
        )

        change = (outcome != outcome.shift(1)).cumsum()
        run_length = outcome.groupby(change).cumcount() + 1
        streak = np.where(outcome == "W", run_length,
                  np.where(outcome == "L", -run_length, 0))

        streak = pd.Series(streak, index=sub.index)
        streak[outcome.isna()] = 0
        return streak

    all_matches["win_loss_streak"] = (
        all_matches.groupby("team", group_keys=False).apply(_streak)
    )

    team_stats = (
        all_matches[["date", "team",
                     "roll_gf", "roll_ga", "roll_won", "roll_drew",
                     "roll_points", "roll_clean_sheet",
                     "form_points", "form_goal_diff",
                     "form_goals_scored", "form_goals_conceded",
                     "win_loss_streak"]]
        .rename(columns={
            "roll_gf":          "avg_goals_scored",
            "roll_ga":          "avg_goals_conceded",
            "roll_won":         "win_rate",
            "roll_drew":        "draw_rate",
            "roll_points":      "avg_points",
            "roll_clean_sheet": "clean_sheet_rate",
        })
    )

    print(f"= Rolling team stats:     {len(team_stats):>7,} rows  "
          f"(long run={window}, current form={form_window})")
    return team_stats


def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative head-to-head record using only prior matches. Specifically:

      h2h_avg_goal_diff    : average goals(home) - goals(away) from home team's perspective 
      h2h_avg_total_goals  : average total goals per H2H meeting
      h2h_days_since_last  : days since the two teams last met. Recent matches are stronger signals than those multiple years ago. Capped at 10 years to prevent extreme outliers
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
                "h2h_home_win_rate":   0.5,
                "h2h_away_win_rate":   0.5,
                "h2h_draw_rate":       0.0,
                "h2h_total":           0,
                "h2h_avg_goal_diff":   0.0,
                "h2h_avg_total_goals": 2.5,
                "h2h_days_since_last": 3650,
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

        home_gf = (
            prior.loc[prior["home_team"] == home, "home_score"].sum()
            + prior.loc[prior["away_team"] == home, "away_score"].sum()
        )
        home_ga = (
            prior.loc[prior["home_team"] == home, "away_score"].sum()
            + prior.loc[prior["away_team"] == home, "home_score"].sum()
        )
        total_goals = home_gf + home_ga

        days_since = (date - prior["date"].max()).days
        days_since = min(days_since, 3650)

        records.append({
            "h2h_home_win_rate":   home_wins / total,
            "h2h_away_win_rate":   away_wins / total,
            "h2h_draw_rate":       draws     / total,
            "h2h_total":           total,
            "h2h_avg_goal_diff":   (home_gf - home_ga) / total,
            "h2h_avg_total_goals": total_goals / total,
            "h2h_days_since_last": days_since,
        })

    print(f"= H2H features computed:  {len(records):>7,} rows")
    return pd.DataFrame(records)


FEATURE_COLS = [
    # Last 10 matches
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate",         "home_draw_rate",
    "home_avg_points",       "home_clean_sheet_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate",         "away_draw_rate",
    "away_avg_points",       "away_clean_sheet_rate",
    # Last 5 matches
    "home_form_points",      "home_form_goal_diff",
    "home_form_goals_scored","home_form_goals_conceded",
    "away_form_points",      "away_form_goal_diff",
    "away_form_goals_scored","away_form_goals_conceded",
    # Win/loss streaks
    "home_win_loss_streak",  "away_win_loss_streak",
    # H2H
    "h2h_home_win_rate",     "h2h_away_win_rate",
    "h2h_draw_rate",         "h2h_total",
    "h2h_avg_goal_diff",     "h2h_avg_total_goals",
    "h2h_days_since_last",
    # FIFA rankings + momentum
    "home_fifa_rank",        "home_fifa_points",
    "away_fifa_rank",        "away_fifa_points",
    "rank_diff",             "points_diff",
    "home_ranking_momentum", "away_ranking_momentum",
    # Match context
    "is_world_cup", "is_neutral", "is_knockout",
]

TARGET_COLS = ["result", "result_encoded", "home_score", "away_score"]


def attach_ranking_momentum(matches: pd.DataFrame,
                             rankings: pd.DataFrame,
                             side: str) -> pd.DataFrame:
    """
    diff.points = total_points - previous_points
    Positive value: the team gained ranking points since the last update i.e. improving
    Negative value: the team dropped ranking points since the last update i.e. declining 
    A team ranked #15 and rising fast gives a different signal from a team ranked #15 and falling
    """
    team_col = f"{side}_team"
    mom_col  = f"{side}_ranking_momentum"

    momentum_lookup = (
        rankings[["rank_date", "country_full", "diff.points"]]
        .rename(columns={"diff.points": "momentum"})
        .sort_values(["country_full", "rank_date"])
        .reset_index(drop=True)
    )

    match_keys = (
        matches[["date", team_col]]
        .copy()
        .rename(columns={team_col: "country_full"})
        .reset_index()
        .sort_values("date")
    )
    match_keys["country_full"] = match_keys["country_full"].replace(
        RESULTS_TO_RANKINGS_ALIASES
    )

    results_list = []
    for team, group in match_keys.groupby("country_full", sort=False):
        team_mom = momentum_lookup[momentum_lookup["country_full"] == team]

        if team_mom.empty:
            group[mom_col] = 0.0
            results_list.append(group[["index", mom_col]])
            continue

        merged = pd.merge_asof(
            group.sort_values("date"),
            team_mom[["rank_date", "momentum"]].rename(
                columns={"rank_date": "date", "momentum": mom_col}
            ),
            on="date",
            direction="backward",
        )
        results_list.append(merged[["index", mom_col]])

    mom_df = pd.concat(results_list).set_index("index").sort_index()
    return matches.join(mom_df)


def build_feature_dataset(matches: pd.DataFrame,
                           team_stats: pd.DataFrame,
                           rankings: pd.DataFrame) -> pd.DataFrame:

    home_stats = (
        team_stats
        .rename(columns={c: f"home_{c}"
                         for c in team_stats.columns
                         if c not in ("date", "team")})
        .rename(columns={"team": "home_team"})
    )
    df = matches.merge(home_stats, on=["date", "home_team"], how="left")

    away_stats = (
        team_stats
        .rename(columns={c: f"away_{c}"
                         for c in team_stats.columns
                         if c not in ("date", "team")})
        .rename(columns={"team": "away_team"})
    )
    df = df.merge(away_stats, on=["date", "away_team"], how="left")

    ranking_lookup = build_ranking_lookup(rankings)
    df = df.sort_values("date").reset_index(drop=True)
    df = attach_rankings(df, ranking_lookup, side="home")
    df = attach_rankings(df, ranking_lookup, side="away")

    df["rank_diff"]   = df["away_fifa_rank"]   - df["home_fifa_rank"]
    df["points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]

    df = attach_ranking_momentum(df, rankings, side="home")
    df = attach_ranking_momentum(df, rankings, side="away")

    print("= Computing H2H features (may take a few minutes on 30 k rows)...")
    h2h = compute_h2h_features(df)
    df  = pd.concat([df.reset_index(drop=True),
                     h2h.reset_index(drop=True)], axis=1)


    GROUP_STAGE_ENDS = {
        1998: "1998-06-26", 2002: "2002-06-14", 2006: "2006-06-23",
        2010: "2010-06-23", 2014: "2014-06-26", 2018: "2018-06-28",
        2022: "2022-12-02", 2026: "2026-06-27",
    }
    is_wc = df["is_world_cup"] == 1
    year  = df["date"].dt.year

    ko_mask = pd.Series(False, index=df.index)
    for wc_year, cutoff in GROUP_STAGE_ENDS.items():
        ko_mask |= (is_wc & (year == wc_year) & (df["date"] > cutoff))

    df["is_knockout"] = ko_mask.astype(int)

    df["result_encoded"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    print(f"= Feature dataset built:  {len(df):>7,} rows x {df.shape[1]} columns")
    return df

def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select feature and target columns and drop rows with NaN features
    """
    keep      = ["date", "home_team", "away_team"] + FEATURE_COLS + TARGET_COLS
    available = [c for c in keep if c in df.columns]
    final     = df[available].dropna(subset=FEATURE_COLS, how="any")

    print(f"= Final dataset:          {len(final):>7,} rows  "
          f"(dropped {len(df) - len(final):,} rows with NaN features)")
    return final



def run_pipeline() -> pd.DataFrame:
    print("\nStarting Data Preparation Pipeline\n" + "=" * 50)

    matches  = load_match_results()
    rankings = load_fifa_rankings()

    matches    = clean_match_results(matches)
    team_stats = compute_team_rolling_stats(matches, window=10)
    features   = build_feature_dataset(matches, team_stats, rankings)
    final      = prepare_final_dataset(features)

    out_path = PROCESSED_DIR / "features.csv"
    final.to_csv(out_path, index=False)
    print(f"\n=== Saved -> {out_path}")

    print("\n=== Feature Summary:")
    print(final[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(3)
          .to_string())

    print("\n=== Target Distribution:")
    print(final["result"].value_counts().to_string())

    return final

if __name__ == "__main__":
    df = run_pipeline()
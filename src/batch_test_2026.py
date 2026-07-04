"""
Batch test the trained models against real 2026 World Cup group stage results.

`data/raw/results.csv` already contains actual final scores for the 2026 group stage. 
This script pulls those 72 matches and re-predicts each one using the trained outcome + score models
Then compares predictions against what actually happened and reports accuracy, MAE, and a full per-match comparison table

Run:
    python src/batch_test_2026.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

sys.path.append(str(BASE_DIR / "src"))

FEATURE_COLS = [
    # Last 10 matches
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate",         "home_draw_rate",
    "home_avg_points",       "home_clean_sheet_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate",         "away_draw_rate",
    "away_avg_points",       "away_clean_sheet_rate",
    # Last 5 matches
    "home_form_points",       "home_form_goal_diff",
    "home_form_goals_scored", "home_form_goals_conceded",
    "away_form_points",       "away_form_goal_diff",
    "away_form_goals_scored", "away_form_goals_conceded",
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

RESULT_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


DRAW_MARGIN_GROUP_STAGE = 0.08
DRAW_MARGIN_KNOCKOUT = 0.00


def apply_draw_margin(proba_row: np.ndarray, classes: list, margin: float) -> str:
    draw_idx = classes.index("D")
    top_idx = int(np.argmax(proba_row))
    if proba_row[draw_idx] >= proba_row[top_idx] - margin:
        return "D"
    return classes[top_idx]

GROUP_STAGE_START = "2026-06-11"
GROUP_STAGE_END = "2026-06-27"

# Knockout round dates are open-ended in case of delays or reschedules
# Current WC Final date is set as 2026-07-20
KNOCKOUT_START = "2026-06-28"
KNOCKOUT_END = "2026-07-31"


def load_stage_features(stage: str = "group") -> pd.DataFrame:
    """
    Pull 2026 World Cup rows out of the features.csv for a given stage
    stage : "group" or "knockout"
    """
    if stage not in ("group", "knockout"):
        raise ValueError(f"stage must be 'group' or 'knockout', got {stage!r}")

    start, end = (
        (GROUP_STAGE_START, GROUP_STAGE_END)
        if stage == "group"
        else (KNOCKOUT_START, KNOCKOUT_END)
    )

    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "features.csv not found"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    mask = (df["date"] >= start) & (df["date"] <= end)
    stage_df = df[mask].copy().reset_index(drop=True)

    label = "group stage" if stage == "group" else "knockout round"
    print(f"Found {len(stage_df)} {label} matches with usable features")

    # Sanity check against raw results to flag anything dropped or unplayed
    raw = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
    raw_wc = raw[
        (raw["tournament"] == "FIFA World Cup")
        & (raw["date"] >= start)
        & (raw["date"] <= end)
    ]
    played = raw_wc.dropna(subset=["home_score", "away_score"])
    dropped = len(played) - len(stage_df)
    if dropped > 0:
        print(
            f"{dropped} {label} matches were excluded "
            f"(likely teams with insufficient history like debutants with <3 prior matches)"
        )
    unplayed = len(raw_wc) - len(played)
    if unplayed > 0:
        print(f"{unplayed} {label} matches not yet played")

    return stage_df


def run_batch_predictions(df: pd.DataFrame, stage: str = "group") -> pd.DataFrame:
    """
    Run the trained models on every row's pre-match features and attach predictions alongside the actual results that are already in the row.
    """
    draw_margin = DRAW_MARGIN_KNOCKOUT if stage == "knockout" else DRAW_MARGIN_GROUP_STAGE

    outcome_model = joblib.load(MODELS_DIR / "outcome_model.pkl")
    score_model = joblib.load(MODELS_DIR / "score_model.pkl")

    X = df[FEATURE_COLS]

    proba = outcome_model.predict_proba(X)
    classes = list(outcome_model.classes_)
    pred_result = np.array([apply_draw_margin(row, classes, margin=draw_margin) for row in proba])

    proba_df = pd.DataFrame(proba, columns=[RESULT_LABELS[c] for c in classes])

    raw_scores = score_model.predict(X)
    raw_scores = np.clip(raw_scores, 0, None)

    results = df[["date", "home_team", "away_team",
                  "home_score", "away_score", "result"]].copy()
    results = results.rename(columns={
        "home_score": "actual_home_score",
        "away_score": "actual_away_score",
        "result": "actual_result",
    })

    results["predicted_result"] = pred_result
    results["expected_home_goals"] = raw_scores[:, 0].round(2)
    results["expected_away_goals"] = raw_scores[:, 1].round(2)
    results["predicted_home_score"] = np.rint(raw_scores[:, 0]).astype(int)
    results["predicted_away_score"] = np.rint(raw_scores[:, 1]).astype(int)

    results = pd.concat([results.reset_index(drop=True), proba_df], axis=1)

    results["outcome_correct"] = results["predicted_result"] == results["actual_result"]
    results["exact_score_correct"] = (
        (results["predicted_home_score"] == results["actual_home_score"])
        & (results["predicted_away_score"] == results["actual_away_score"])
    )
    return results


def print_report(results: pd.DataFrame, stage: str = "group"):
    label = "Group Stage" if stage == "group" else "Knockout Round"
    n = len(results)
    outcome_acc = results["outcome_correct"].mean()
    exact_score_acc = results["exact_score_correct"].mean()

    home_mae = (results["expected_home_goals"] - results["actual_home_score"]).abs().mean()
    away_mae = (results["expected_away_goals"] - results["actual_away_score"]).abs().mean()

    print("\n" + "=" * 65)
    print(f"BATCH TEST REPORT — 2026 World Cup {label} ({n} matches)")
    print("=" * 65)
    print(f"Outcome accuracy (H/D/A):      {outcome_acc:.1%}  ({results['outcome_correct'].sum()}/{n})")
    print(f"Exact scoreline accuracy:      {exact_score_acc:.1%}  ({results['exact_score_correct'].sum()}/{n})")
    print(f"Home goals MAE:                {home_mae:.3f}")
    print(f"Away goals MAE:                {away_mae:.3f}")

    print("\n=== Accuracy by actual outcome type")
    by_type = results.groupby("actual_result")["outcome_correct"].agg(["mean", "count"])
    by_type.index = by_type.index.map(RESULT_LABELS)
    by_type.columns = ["accuracy", "n_matches"]
    print(by_type.round(3).to_string())

    print("\n=== Confusion: predicted vs actual")
    confusion = pd.crosstab(
        results["actual_result"].map(RESULT_LABELS),
        results["predicted_result"].map(RESULT_LABELS),
        rownames=["Actual"], colnames=["Predicted"]
    )
    print(confusion.to_string())

    print("\n=== First 20 match results")
    display_cols = [
        "date", "home_team", "away_team",
        "actual_home_score", "actual_away_score",
        "predicted_home_score", "predicted_away_score",
        "actual_result", "predicted_result", "outcome_correct",
    ]
    preview = results[display_cols].copy()
    preview["date"] = preview["date"].dt.date
    print(preview.head(20).to_string(index=False))

    print("\n=== Biggest misses i.e. largest goal error) ")
    results["total_goal_error"] = (
        (results["expected_home_goals"] - results["actual_home_score"]).abs()
        + (results["expected_away_goals"] - results["actual_away_score"]).abs()
    )
    worst = results.sort_values("total_goal_error", ascending=False).head(10)
    print(worst[["date", "home_team", "away_team",
                 "actual_home_score", "actual_away_score",
                 "expected_home_goals", "expected_away_goals"]].to_string(index=False))


def main(stage: str = "group"):
    df = load_stage_features(stage)

    if df.empty:
        print(f"\nNo played {stage} matches with usable features yet hence nothing to test.")
        return

    results = run_batch_predictions(df, stage=stage)

    out_path = PROCESSED_DIR / f"{stage}_2026_predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"\n Full results saved -> {out_path}")

    print_report(results, stage=stage)


if __name__ == "__main__":
    # Run both stages by default
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("group", "knockout"):
        main(sys.argv[1])
    else:
        print("\n" + "=" * 65)
        print("GROUP STAGE")
        print("=" * 65)
        main("group")

        print("\n" + "=" * 65)
        print("KNOCKOUT ROUND")
        print("=" * 65)
        main("knockout")
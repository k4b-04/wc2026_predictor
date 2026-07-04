"""
Walk-forward calibration check for the DRAW_MARGIN constants in predictor.py

Run:
    python src/margin_calibration.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLS = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate",         "home_draw_rate",
    "home_avg_points",       "home_clean_sheet_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate",         "away_draw_rate",
    "away_avg_points",       "away_clean_sheet_rate",
    "home_form_points",       "home_form_goal_diff",
    "home_form_goals_scored", "home_form_goals_conceded",
    "away_form_points",       "away_form_goal_diff",
    "away_form_goals_scored", "away_form_goals_conceded",
    "home_win_loss_streak",  "away_win_loss_streak",
    "h2h_home_win_rate",     "h2h_away_win_rate",
    "h2h_draw_rate",         "h2h_total",
    "h2h_avg_goal_diff",     "h2h_avg_total_goals",
    "h2h_days_since_last",
    "home_fifa_rank",        "home_fifa_points",
    "away_fifa_rank",        "away_fifa_points",
    "rank_diff",             "points_diff",
    "home_ranking_momentum", "away_ranking_momentum",
    "is_world_cup", "is_neutral", "is_knockout",
]

MARGIN_GRID = [round(m, 2) for m in np.arange(0.00, 0.22, 0.02)]

try:
    BEST_PARAMS = eval((MODELS_DIR / "best_params.txt").read_text())
except FileNotFoundError:
    BEST_PARAMS = {"max_depth": 8, "min_samples_leaf": 5, "n_estimators": 200}

CURRENT_GROUP_MARGIN = 0.10
CURRENT_KNOCKOUT_MARGIN = 0.06


KNOCKOUT_STARTS = {
    1998: "1998-06-27", 2002: "2002-06-15", 2006: "2006-06-24",
    2010: "2010-06-24", 2014: "2014-06-27", 2018: "2018-06-29",
    2022: "2022-12-03",
}


def fit_rf(X, y):
    return RandomForestClassifier(
        **BEST_PARAMS, class_weight="balanced", random_state=42, n_jobs=1
    ).fit(X, y)


def sweep_margins(proba: np.ndarray, classes: list, y_true: pd.Series) -> pd.DataFrame:
    draw_idx = classes.index("D")
    top_idx = np.argmax(proba, axis=1)
    draw_prob = proba[:, draw_idx]
    top_prob = proba[np.arange(len(proba)), top_idx]

    rows = []
    y_true_arr = y_true.values
    n_draws = (y_true_arr == "D").sum()

    for margin in MARGIN_GRID:
        use_draw = draw_prob >= (top_prob - margin)
        preds = np.where(use_draw, "D", np.array(classes)[top_idx])

        acc = (preds == y_true_arr).mean()
        f1s = []
        for cls in ["H", "D", "A"]:
            tp = ((preds == cls) & (y_true_arr == cls)).sum()
            fp = ((preds == cls) & (y_true_arr != cls)).sum()
            fn = ((preds != cls) & (y_true_arr == cls)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = np.mean(f1s)

        draw_recall = (
            ((preds == "D") & (y_true_arr == "D")).sum() / n_draws if n_draws > 0 else np.nan
        )
        draws_predicted = int((preds == "D").sum())

        rows.append({
            "margin": margin,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "draw_recall": round(draw_recall, 4) if not np.isnan(draw_recall) else np.nan,
            "draws_predicted": draws_predicted,
            "n": len(y_true_arr),
            "n_draws": int(n_draws),
        })

    return pd.DataFrame(rows)


def analyze_group_margin(df: pd.DataFrame, test_years=range(2019, 2027)) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("ANALYSIS 1: GROUP-STAGE MARGIN -- walk-forward across 8 independent years")
    print("(this margin is applied to ~everything: friendlies, qualifiers, AND WC group games)")
    print("=" * 72)

    all_sweeps = []
    for year in test_years:
        train_df = df[df["date"].dt.year < year]
        test_df = df[df["date"].dt.year == year]
        if len(test_df) == 0 or len(train_df) < 200:
            continue

        model = fit_rf(train_df[FEATURE_COLS], train_df["result"])
        proba = model.predict_proba(test_df[FEATURE_COLS])
        classes = list(model.classes_)

        sweep = sweep_margins(proba, classes, test_df["result"])
        sweep["fold"] = year
        all_sweeps.append(sweep)

        best_row = sweep.loc[sweep["macro_f1"].idxmax()]
        current_row = sweep.loc[sweep["margin"] == CURRENT_GROUP_MARGIN].iloc[0]
        print(f"\n  Year {year} (n={len(test_df)}, {int((test_df['result']=='D').sum())} draws):"
              f"  best margin={best_row['margin']:.2f} (macro_f1={best_row['macro_f1']:.3f}, "
              f"acc={best_row['accuracy']:.3f})  |  "
              f"at margin=0.10: macro_f1={current_row['macro_f1']:.3f}, acc={current_row['accuracy']:.3f}, "
              f"draw_recall={current_row['draw_recall']:.3f}")

    full = pd.concat(all_sweeps, ignore_index=True)

    print("\n  --- Aggregated across all 8 years, per margin ---")
    agg = full.groupby("margin").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_draw_recall=("draw_recall", "mean"),
        std_macro_f1=("macro_f1", "std"),
    ).round(4)
    print(agg.to_string())

    best_margins_per_fold = full.loc[full.groupby("fold")["macro_f1"].idxmax()][["fold", "margin", "macro_f1"]]
    print("\n  --- Best margin PER FOLD (what would look 'optimal' if you only saw that year) ---")
    print(best_margins_per_fold.to_string(index=False))
    print(f"\n  Spread of per-fold optimal margins: min={best_margins_per_fold['margin'].min()}, "
          f"max={best_margins_per_fold['margin'].max()}, "
          f"mode={best_margins_per_fold['margin'].mode().tolist()}")

    return full


def analyze_knockout_margin(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("ANALYSIS 2: KNOCKOUT MARGIN -- leave-one-tournament-out (7 World Cups, 1998-2022)")
    print("(only 119 historical knockout matches total exist in the data -- ~16-24 per fold)")
    print("=" * 72)

    all_sweeps = []
    for wc_year, ko_start in KNOCKOUT_STARTS.items():
        train_df = df[df["date"] < ko_start]
        test_df = df[
            (df["is_knockout"] == 1) &
            (df["date"].dt.year == wc_year) &
            (df["date"] >= ko_start)
        ]
        if len(test_df) == 0:
            continue

        model = fit_rf(train_df[FEATURE_COLS], train_df["result"])
        proba = model.predict_proba(test_df[FEATURE_COLS])
        classes = list(model.classes_)

        sweep = sweep_margins(proba, classes, test_df["result"])
        sweep["fold"] = wc_year
        all_sweeps.append(sweep)

        best_row = sweep.loc[sweep["macro_f1"].idxmax()]
        current_row = sweep.loc[sweep["margin"] == CURRENT_KNOCKOUT_MARGIN].iloc[0]
        print(f"\n  WC {wc_year} knockouts (n={len(test_df)}, {int((test_df['result']=='D').sum())} draws):"
              f"  best margin={best_row['margin']:.2f} (macro_f1={best_row['macro_f1']:.3f})  |  "
              f"at margin=0.06: macro_f1={current_row['macro_f1']:.3f}, "
              f"draw_recall={current_row['draw_recall']:.3f}, draws_predicted={current_row['draws_predicted']}")

    full = pd.concat(all_sweeps, ignore_index=True)

    print("\n  --- Aggregated across all 7 tournaments, per margin ---")
    agg = full.groupby("margin").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_draw_recall=("draw_recall", "mean"),
        std_macro_f1=("macro_f1", "std"),
    ).round(4)
    print(agg.to_string())

    best_margins_per_fold = full.loc[full.groupby("fold")["macro_f1"].idxmax()][["fold", "margin", "macro_f1"]]
    print("\n  --- Best margin PER TOURNAMENT ---")
    print(best_margins_per_fold.to_string(index=False))
    print(f"\n  Spread of per-tournament optimal margins: min={best_margins_per_fold['margin'].min()}, "
          f"max={best_margins_per_fold['margin'].max()}, "
          f"mode={best_margins_per_fold['margin'].mode().tolist()}")

    return full


def main():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    group_sweep = analyze_group_margin(df)
    knockout_sweep = analyze_knockout_margin(df)

    group_sweep.to_csv(PROCESSED_DIR / "margin_calibration_group.csv", index=False)
    knockout_sweep.to_csv(PROCESSED_DIR / "margin_calibration_knockout.csv", index=False)

    print("\n" + "=" * 72)
    print("Saved: data/processed/margin_calibration_group.csv")
    print("Saved: data/processed/margin_calibration_knockout.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
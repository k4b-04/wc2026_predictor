"""
Phase 2: Model Training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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

# Walk-forward test years: 2019 is the first year with a meaningful amount of training history behind it
WALK_FORWARD_YEARS = list(range(2019, 2027))

PARAM_GRID = {
    "n_estimators":     [200, 350],
    "max_depth":         [8, 14],
    "min_samples_leaf":  [5, 10],
}


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"= Loaded features: {len(df):,} rows x {df.shape[1]} columns")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    return df


def tune_outcome_model(df: pd.DataFrame, tuning_cutoff_year: int = 2019):
    tuning_pool = df[df["date"].dt.year < tuning_cutoff_year].copy()
    X = tuning_pool[FEATURE_COLS]
    y = tuning_pool["result"]

    print(f"\n= Hyperparameter search (GridSearchCV + TimeSeriesSplit)")
    print(f"Tuning pool: {len(tuning_pool):,} rows, all dated before {tuning_cutoff_year}")
    print(f"Grid size: {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_leaf'])} combos x 5 folds")

    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    search = GridSearchCV(
        base_model,
        PARAM_GRID,
        cv=tscv,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV macro-F1 (avg across {n_splits} time-ordered folds): {search.best_score_:.4f}")

    cv_results = pd.DataFrame(search.cv_results_)
    best_row = cv_results.loc[cv_results["rank_test_score"] == 1].iloc[0]
    fold_scores = [best_row[f"split{i}_test_score"] for i in range(n_splits)]
    print(f"Per-fold macro-F1 for best params: {[round(s, 3) for s in fold_scores]}")
    print(f"  (std dev across folds: {np.std(fold_scores):.4f} -- lower means more stable across time periods)")

    return search.best_params_


def walk_forward_evaluate(df: pd.DataFrame, best_params: dict, test_years=WALK_FORWARD_YEARS):
    rows = []
    for year in test_years:
        train_df = df[df["date"].dt.year < year]
        test_df = df[df["date"].dt.year == year]

        if len(test_df) == 0 or len(train_df) < 200:
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df["result"]
        X_test, y_test = test_df[FEATURE_COLS], test_df["result"]

        model = RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")

        draw_mask = y_test == "D"
        draw_recall = (
            (preds[draw_mask.values] == "D").mean() if draw_mask.sum() > 0 else np.nan
        )

        rows.append({
            "test_year": year,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_draws_in_test": int(draw_mask.sum()),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "draw_recall": round(draw_recall, 4) if not np.isnan(draw_recall) else np.nan,
        })

    results = pd.DataFrame(rows)

    print("\n= Walk-Forward Evaluation (expanding window, one fold per year)")
    print("=" * 70)
    print(results.to_string(index=False))
    print("\nSummary across all test years:")
    print(f"  Accuracy:  mean={results['accuracy'].mean():.4f}  std={results['accuracy'].std():.4f}  "
          f"min={results['accuracy'].min():.4f}  max={results['accuracy'].max():.4f}")
    print(f"  Macro F1:  mean={results['macro_f1'].mean():.4f}  std={results['macro_f1'].std():.4f}  "
          f"min={results['macro_f1'].min():.4f}  max={results['macro_f1'].max():.4f}")
    print(f"  Draw recall: mean={results['draw_recall'].mean():.4f}  std={results['draw_recall'].std():.4f}  "
          f"min={results['draw_recall'].min():.4f}  max={results['draw_recall'].max():.4f}")

    return results


def evaluate_outcome_model(model, X_test, y_test, label="Final Holdout (2018+)"):
    preds = model.predict(X_test)

    print(f"\n= Outcome Model Evaluation -- {label}")
    print("=" * 50)
    print(f"Accuracy:    {accuracy_score(y_test, preds):.4f}")
    print(f"Macro F1:    {f1_score(y_test, preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    labels = ["H", "D", "A"]
    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{x}" for x in labels],
                            columns=[f"Pred_{x}" for x in labels])
    print(cm_df)

    return preds


def train_score_model(X_train, y_train):
    base_regressor = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, y_train)
    return model


def evaluate_score_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    home_pred = preds[:, 0]
    away_pred = preds[:, 1]
    home_true = y_test["home_score"].values
    away_true = y_test["away_score"].values

    print("\n= Score Model Evaluation")
    print("=" * 50)

    home_mae = mean_absolute_error(home_true, home_pred)
    away_mae = mean_absolute_error(away_true, away_pred)
    overall_mae = mean_absolute_error(y_test.values, preds)

    print(f"Home goals MAE:    {home_mae:.4f}")
    print(f"Away goals MAE:    {away_mae:.4f}")
    print(f"Overall score MAE: {overall_mae:.4f}")

    rmse = np.sqrt(mean_squared_error(y_test.values, preds))
    print(f"Overall RMSE:      {rmse:.4f}")
    print(f"R² Score:          {r2_score(y_test.values, preds):.4f}")

    rounded = np.rint(preds).astype(int)
    exact_score_acc = np.mean(
        (rounded[:, 0] == home_true) &
        (rounded[:, 1] == away_true)
    )
    print(f"Exact score accuracy after rounding: {exact_score_acc:.4f}")

    return preds


def show_feature_importance(model):
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n= Top 15 Outcome Feature Importances")
    print("=" * 50)
    print(fi.head(15).to_string(index=False))

    return fi


def save_models(outcome_model, score_model, feature_importance, best_params, walk_forward_results):
    outcome_path = MODELS_DIR / "outcome_model.pkl"
    score_path = MODELS_DIR / "score_model.pkl"
    fi_path = MODELS_DIR / "feature_importance.csv"
    wf_path = MODELS_DIR / "walk_forward_results.csv"
    params_path = MODELS_DIR / "best_params.txt"

    joblib.dump(outcome_model, outcome_path)
    joblib.dump(score_model, score_path)
    feature_importance.to_csv(fi_path, index=False)
    walk_forward_results.to_csv(wf_path, index=False)
    params_path.write_text(str(best_params))

    print("\n= Saved models")
    print(f"Outcome model      -> {outcome_path}")
    print(f"Score model        -> {score_path}")
    print(f"Importance         -> {fi_path}")
    print(f"Walk-forward table -> {wf_path}")
    print(f"Best params        -> {params_path}")


def run_training():
    print("\nStarting Model Training Pipeline")
    print("=" * 50)

    df = load_features()

    best_params = tune_outcome_model(df, tuning_cutoff_year=2019)

    wf_results = walk_forward_evaluate(df, best_params, test_years=WALK_FORWARD_YEARS)

    train_df = df[df["date"].dt.year < 2018]
    test_df = df[df["date"].dt.year >= 2018]

    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]
    y_train_outcome, y_test_outcome = train_df["result"], test_df["result"]
    y_train_score, y_test_score = train_df[["home_score", "away_score"]], test_df[["home_score", "away_score"]]

    print(f"\n= Training final outcome classifier with tuned params: {best_params}")
    outcome_model = RandomForestClassifier(
        **best_params, class_weight="balanced", random_state=42, n_jobs=1,
    )
    outcome_model.fit(X_train, y_train_outcome)
    evaluate_outcome_model(outcome_model, X_test, y_test_outcome, label="Final Holdout (2018+, tuned params)")

    feature_importance = show_feature_importance(outcome_model)

    print("\n= Training score regressor...")
    score_model = train_score_model(X_train, y_train_score)
    evaluate_score_model(score_model, X_test, y_test_score)

    print("\n= Refitting final deployed model on the FULL dataset (all years)...")
    deployed_outcome_model = RandomForestClassifier(
        **best_params, class_weight="balanced", random_state=42, n_jobs=1,
    )
    deployed_outcome_model.fit(df[FEATURE_COLS], df["result"])

    deployed_score_model = train_score_model(df[FEATURE_COLS], df[["home_score", "away_score"]])

    save_models(deployed_outcome_model, deployed_score_model, feature_importance, best_params, wf_results)

    print("\n= Model training complete!")

    return deployed_outcome_model, deployed_score_model


if __name__ == "__main__":
    run_training()
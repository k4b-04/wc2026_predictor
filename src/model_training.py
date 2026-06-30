"""
Phase 2: Model Training
FIFA World Cup 2026 Match Outcome Predictor

Trains:
- Match outcome classifier: H / D / A
- Score prediction model: home_score and away_score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLS = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate", "home_draw_rate",
    "home_avg_points", "home_clean_sheet_rate",

    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate", "away_draw_rate",
    "away_avg_points", "away_clean_sheet_rate",

    "h2h_home_win_rate", "h2h_away_win_rate",
    "h2h_draw_rate", "h2h_total",

    "home_fifa_rank", "home_fifa_points",
    "away_fifa_rank", "away_fifa_points",
    "rank_diff", "points_diff",

    "is_world_cup", "is_neutral",
]

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    print(f"✅ Loaded features: {len(df):,} rows x {df.shape} columns")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    return df

# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────

def time_based_split(df: pd.DataFrame, cutoff_year: int = 2018):
    """
    Uses time-based validation to avoid future data leaking into past predictions.

    Train: matches before cutoff_year
    Test: matches from cutoff_year onward
    """
    train_df = df[df["date"].dt.year < cutoff_year].copy()
    test_df = df[df["date"].dt.year >= cutoff_year].copy()

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]

    y_train_outcome = train_df["result"]
    y_test_outcome = test_df["result"]

    y_train_score = train_df[["home_score", "away_score"]]
    y_test_score = test_df[["home_score", "away_score"]]

    print("\n📦 Time-based split")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    return X_train, X_test, y_train_outcome, y_test_outcome, y_train_score, y_test_score

# ─────────────────────────────────────────
# OUTCOME MODEL
# ─────────────────────────────────────────

def train_outcome_model(X_train, y_train):
    """
    Train classifier for H/D/A outcome.
    """
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def evaluate_outcome_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\n🎯 Outcome Model Evaluation")
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

# ─────────────────────────────────────────
# SCORE MODEL
# ─────────────────────────────────────────

def train_score_model(X_train, y_train):
    """
    Train model to predict [home_score, away_score].
    """
    base_regressor = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, y_train)

    return model

def evaluate_score_model(model, X_test, y_test):
    preds = model.predict(X_test)

    # Football scores should be non-negative
    preds = np.clip(preds, 0, None)

    home_pred = preds[:, 0]
    away_pred = preds[:, 1]

    home_true = y_test["home_score"].values
    away_true = y_test["away_score"].values

    print("\n⚽ Score Model Evaluation")
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

    # Rounded score accuracy
    rounded = np.rint(preds).astype(int)
    exact_score_acc = np.mean(
        (rounded[:, 0] == home_true) &
        (rounded[:, 1] == away_true)
    )

    print(f"Exact score accuracy after rounding: {exact_score_acc:.4f}")

    return preds

# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────

def show_feature_importance(model):
    importances = model.feature_importances_

    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n📊 Top 15 Outcome Feature Importances")
    print("=" * 50)
    print(fi.head(15).to_string(index=False))

    return fi

# ─────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────

def save_models(outcome_model, score_model, feature_importance):
    outcome_path = MODELS_DIR / "outcome_model.pkl"
    score_path = MODELS_DIR / "score_model.pkl"
    fi_path = MODELS_DIR / "feature_importance.csv"

    joblib.dump(outcome_model, outcome_path)
    joblib.dump(score_model, score_path)
    feature_importance.to_csv(fi_path, index=False)

    print("\n💾 Saved models")
    print(f"Outcome model -> {outcome_path}")
    print(f"Score model   -> {score_path}")
    print(f"Importance    -> {fi_path}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_training():
    print("\n🚀 Starting Model Training Pipeline")
    print("=" * 50)

    df = load_features()

    X_train, X_test, y_train_outcome, y_test_outcome, y_train_score, y_test_score = (
        time_based_split(df, cutoff_year=2018)
    )

    print("\n🧠 Training outcome classifier...")
    outcome_model = train_outcome_model(X_train, y_train_outcome)
    evaluate_outcome_model(outcome_model, X_test, y_test_outcome)

    feature_importance = show_feature_importance(outcome_model)

    print("\n🧠 Training score regressor...")
    score_model = train_score_model(X_train, y_train_score)
    evaluate_score_model(score_model, X_test, y_test_score)

    save_models(outcome_model, score_model, feature_importance)

    print("\n✅ Model training complete!")

    return outcome_model, score_model

if __name__ == "__main__":
    run_training()

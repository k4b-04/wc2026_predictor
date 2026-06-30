"""
Prediction logic for FIFA World Cup 2026 Match Outcome Predictor.

Loads trained models and creates prediction rows for future matches.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

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

RESULT_LABELS = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

class WorldCupPredictor:
    def __init__(self):
        self.features = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
        self.outcome_model = joblib.load(MODELS_DIR / "outcome_model.pkl")
        self.score_model = joblib.load(MODELS_DIR / "score_model.pkl")

        self.teams = sorted(
            set(self.features["home_team"].unique()).union(
                set(self.features["away_team"].unique())
            )
        )

    def get_teams(self):
        return self.teams

    def get_latest_team_stats(self, team: str, prefix: str) -> dict:
        """
        Get latest available rolling stats and FIFA ranking info for a team.

        prefix should be 'home' or 'away'.
        """
        home_rows = self.features[self.features["home_team"] == team].copy()
        away_rows = self.features[self.features["away_team"] == team].copy()

        records = []

        if not home_rows.empty:
            for _, row in home_rows.iterrows():
                records.append({
                    "date": row["date"],
                    "avg_goals_scored": row["home_avg_goals_scored"],
                    "avg_goals_conceded": row["home_avg_goals_conceded"],
                    "win_rate": row["home_win_rate"],
                    "draw_rate": row["home_draw_rate"],
                    "avg_points": row["home_avg_points"],
                    "clean_sheet_rate": row["home_clean_sheet_rate"],
                    "fifa_rank": row["home_fifa_rank"],
                    "fifa_points": row["home_fifa_points"],
                })

        if not away_rows.empty:
            for _, row in away_rows.iterrows():
                records.append({
                    "date": row["date"],
                    "avg_goals_scored": row["away_avg_goals_scored"],
                    "avg_goals_conceded": row["away_avg_goals_conceded"],
                    "win_rate": row["away_win_rate"],
                    "draw_rate": row["away_draw_rate"],
                    "avg_points": row["away_avg_points"],
                    "clean_sheet_rate": row["away_clean_sheet_rate"],
                    "fifa_rank": row["away_fifa_rank"],
                    "fifa_points": row["away_fifa_points"],
                })

        if not records:
            raise ValueError(f"No historical data found for team: {team}")

        latest = pd.DataFrame(records).sort_values("date").iloc[-1]

        return {
            f"{prefix}_avg_goals_scored": latest["avg_goals_scored"],
            f"{prefix}_avg_goals_conceded": latest["avg_goals_conceded"],
            f"{prefix}_win_rate": latest["win_rate"],
            f"{prefix}_draw_rate": latest["draw_rate"],
            f"{prefix}_avg_points": latest["avg_points"],
            f"{prefix}_clean_sheet_rate": latest["clean_sheet_rate"],
            f"{prefix}_fifa_rank": latest["fifa_rank"],
            f"{prefix}_fifa_points": latest["fifa_points"],
        }

    def get_h2h_stats(self, home_team: str, away_team: str) -> dict:
        """
        Approximate latest H2H stats from existing feature rows.
        """
        direct = self.features[
            (self.features["home_team"] == home_team) &
            (self.features["away_team"] == away_team)
        ].copy()

        reverse = self.features[
            (self.features["home_team"] == away_team) &
            (self.features["away_team"] == home_team)
        ].copy()

        if direct.empty and reverse.empty:
            return {
                "h2h_home_win_rate": 0.5,
                "h2h_away_win_rate": 0.5,
                "h2h_draw_rate": 0.0,
                "h2h_total": 0,
            }

        candidates = []

        if not direct.empty:
            latest_direct = direct.sort_values("date").iloc[-1]
            candidates.append({
                "date": latest_direct["date"],
                "h2h_home_win_rate": latest_direct["h2h_home_win_rate"],
                "h2h_away_win_rate": latest_direct["h2h_away_win_rate"],
                "h2h_draw_rate": latest_direct["h2h_draw_rate"],
                "h2h_total": latest_direct["h2h_total"],
            })

        if not reverse.empty:
            latest_reverse = reverse.sort_values("date").iloc[-1]

            # Reverse orientation because selected home/away is opposite
            candidates.append({
                "date": latest_reverse["date"],
                "h2h_home_win_rate": latest_reverse["h2h_away_win_rate"],
                "h2h_away_win_rate": latest_reverse["h2h_home_win_rate"],
                "h2h_draw_rate": latest_reverse["h2h_draw_rate"],
                "h2h_total": latest_reverse["h2h_total"],
            })

        latest = sorted(candidates, key=lambda x: x["date"])[-1]
        latest.pop("date")

        return latest

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        is_world_cup: int = 1,
        is_neutral: int = 1
    ) -> pd.DataFrame:
        """
        Build one ML-ready feature row for a selected match.
        """
        if home_team == away_team:
            raise ValueError("Home team and away team cannot be the same.")

        row = {}

        row.update(self.get_latest_team_stats(home_team, "home"))
        row.update(self.get_latest_team_stats(away_team, "away"))
        row.update(self.get_h2h_stats(home_team, away_team))

        row["rank_diff"] = row["away_fifa_rank"] - row["home_fifa_rank"]
        row["points_diff"] = row["home_fifa_points"] - row["away_fifa_points"]

        row["is_world_cup"] = int(is_world_cup)
        row["is_neutral"] = int(is_neutral)

        X = pd.DataFrame([row])

        # Ensure exact feature order
        X = X[FEATURE_COLS]

        return X

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_world_cup: int = 1,
        is_neutral: int = 1
    ) -> dict:
        """
        Predict outcome and score for a match.
        """
        X = self.build_match_features(
            home_team=home_team,
            away_team=away_team,
            is_world_cup=is_world_cup,
            is_neutral=is_neutral
        )

        # ----------------------------
        # Outcome prediction
        # ----------------------------
        outcome_pred = self.outcome_model.predict(X)
        outcome_pred_value = outcome_pred[0] if isinstance(outcome_pred, np.ndarray) else outcome_pred

        # ----------------------------
        # Probability prediction
        # ----------------------------
        if hasattr(self.outcome_model, "predict_proba"):
            raw_proba = self.outcome_model.predict_proba(X)
            classes = self.outcome_model.classes_

            probabilities = np.asarray(raw_proba)

            # Normal sklearn shape: (1, n_classes)
            if probabilities.ndim == 2:
                probabilities = probabilities

            # Safety flatten
            probabilities = probabilities.reshape(-1)

            prob_dict = {}
            for cls, prob in zip(classes, probabilities):
                prob_dict[RESULT_LABELS.get(cls, str(cls))] = float(prob)
        else:
            prob_dict = {}

        # ----------------------------
        # Score prediction
        # ----------------------------
        raw_score_pred = self.score_model.predict(X)

        # Convert model output to flat array: [home_goals, away_goals]
        score_pred = np.asarray(raw_score_pred).reshape(-1)

        if len(score_pred) < 2:
            raise ValueError(
                f"Score model returned unexpected output: {raw_score_pred}"
            )

        # Keep goals non-negative
        score_pred = np.clip(score_pred, 0, None)

        expected_home_goals = float(score_pred[0])
        expected_away_goals = float(score_pred[1])

        # ----------------------------
        # Determine predicted outcome from probabilities
        # ----------------------------
        most_likely_outcome = max(prob_dict, key=prob_dict.get)
        
        # Round scores based on most likely outcome
        if most_likely_outcome == "Home Win":
            # Ensure home team scores more than away team
            predicted_home_goals = max(1, int(round(expected_home_goals)))
            predicted_away_goals = max(0, int(round(expected_away_goals)))
            
            # If rounding gives a draw or away win, adjust
            if predicted_home_goals <= predicted_away_goals:
                predicted_home_goals = predicted_away_goals + 1
                
        elif most_likely_outcome == "Away Win":
            # Ensure away team scores more than home team
            predicted_home_goals = max(0, int(round(expected_home_goals)))
            predicted_away_goals = max(1, int(round(expected_away_goals)))
            
            # If rounding gives a draw or home win, adjust
            if predicted_away_goals <= predicted_home_goals:
                predicted_away_goals = predicted_home_goals + 1
                
        else:  # Draw
            # Ensure both teams score the same
            avg_goals = int(round((expected_home_goals + expected_away_goals) / 2))
            predicted_home_goals = max(0, avg_goals)
            predicted_away_goals = predicted_home_goals

        return {
            "home_team": home_team,
            "away_team": away_team,
            "predicted_result_code": outcome_pred_value,
            "predicted_result": RESULT_LABELS.get(outcome_pred_value, str(outcome_pred_value)),
            "most_likely_outcome": most_likely_outcome,
            "probabilities": prob_dict,
            "expected_home_goals": expected_home_goals,
            "expected_away_goals": expected_away_goals,
            "predicted_home_goals": predicted_home_goals,
            "predicted_away_goals": predicted_away_goals,
            "feature_row": X,
        }


if __name__ == "__main__":
    predictor = WorldCupPredictor()

    # Test prediction
    result = predictor.predict_match("Brazil", "France")

    print("\n🔮 Prediction")
    print("=" * 50)
    print(f"{result['home_team']} vs {result['away_team']}")
    print(f"Most likely outcome: {result['most_likely_outcome']}")
    print(f"Predicted result: {result['predicted_result']}")
    print(f"Expected goals: {result['home_team']} {result['expected_home_goals']:.2f} - {result['expected_away_goals']:.2f} {result['away_team']}")
    print(f"Predicted score: {result['home_team']} {result['predicted_home_goals']} - {result['predicted_away_goals']} {result['away_team']}")

    print("\nProbabilities:")
    for label, prob in result["probabilities"].items():
        print(f"{label}: {prob:.2%}")

"""
Prediction logic: Loads trained models and creates prediction rows for future matches.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLS = [
    # Rolling form (last 10 matches)
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate",         "home_draw_rate",
    "home_avg_points",       "home_clean_sheet_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate",         "away_draw_rate",
    "away_avg_points",       "away_clean_sheet_rate",
    # Current form (last 5 matches)
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
    # FIFA rankings, momentum
    "home_fifa_rank",        "home_fifa_points",
    "away_fifa_rank",        "away_fifa_points",
    "rank_diff",             "points_diff",
    "home_ranking_momentum", "away_ranking_momentum",
    # Match context
    "is_world_cup", "is_neutral", "is_knockout",
]

RESULT_LABELS = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}


# RandomForest.predict() always picks the highest-probability class, which is rarely a draw
# But, draws are genuinely common at around 24.7% of group stage matches
# The draw margin corrects this without having to retrain the model

# Probability margin for group stage matches: Draws get a 10% headroom
DRAW_MARGIN_GROUP_STAGE = 0.10

# Probability margin for knockout matches: Draws get a 6% headroom
# Draws are less common in knockouts given their conntext, at around 21.4%
# Technically, knockout games never draw, but this model is predicting the 90-minute scoreline, which can end in a draw and lead to penalties
DRAW_MARGIN_KNOCKOUT = 0.06


def apply_draw_margin(proba_row: np.ndarray, classes: list, margin: float) -> str:
    """
    Pick the predicted class from a single row of probabilities, boosting Draw if it's within `margin` of the top predicted class
    """
    draw_idx = classes.index("D")
    top_idx = int(np.argmax(proba_row))
    if proba_row[draw_idx] >= proba_row[top_idx] - margin:
        return "D"
    return classes[top_idx]

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
        Get the most recent rolling stats + ranking info for a team to find the latest date
        """
        home_rows = self.features[self.features["home_team"] == team]
        away_rows = self.features[self.features["away_team"] == team]

        records = []

        if not home_rows.empty:
            for _, row in home_rows.iterrows():
                records.append({
                    "date":                row["date"],
                    "avg_goals_scored":    row["home_avg_goals_scored"],
                    "avg_goals_conceded":  row["home_avg_goals_conceded"],
                    "win_rate":            row["home_win_rate"],
                    "draw_rate":           row["home_draw_rate"],
                    "avg_points":          row["home_avg_points"],
                    "clean_sheet_rate":    row["home_clean_sheet_rate"],
                    "form_points":         row["home_form_points"],
                    "form_goal_diff":      row["home_form_goal_diff"],
                    "form_goals_scored":   row["home_form_goals_scored"],
                    "form_goals_conceded": row["home_form_goals_conceded"],
                    "win_loss_streak":     row["home_win_loss_streak"],
                    "fifa_rank":           row["home_fifa_rank"],
                    "fifa_points":         row["home_fifa_points"],
                    "ranking_momentum":    row["home_ranking_momentum"],
                })

        if not away_rows.empty:
            for _, row in away_rows.iterrows():
                records.append({
                    "date":                row["date"],
                    "avg_goals_scored":    row["away_avg_goals_scored"],
                    "avg_goals_conceded":  row["away_avg_goals_conceded"],
                    "win_rate":            row["away_win_rate"],
                    "draw_rate":           row["away_draw_rate"],
                    "avg_points":          row["away_avg_points"],
                    "clean_sheet_rate":    row["away_clean_sheet_rate"],
                    "form_points":         row["away_form_points"],
                    "form_goal_diff":      row["away_form_goal_diff"],
                    "form_goals_scored":   row["away_form_goals_scored"],
                    "form_goals_conceded": row["away_form_goals_conceded"],
                    "win_loss_streak":     row["away_win_loss_streak"],
                    "fifa_rank":           row["away_fifa_rank"],
                    "fifa_points":         row["away_fifa_points"],
                    "ranking_momentum":    row["away_ranking_momentum"],
                })

        if not records:
            raise ValueError(f"No historical data found for team: {team}")

        latest = pd.DataFrame(records).sort_values("date").iloc[-1]

        return {
            f"{prefix}_avg_goals_scored":    latest["avg_goals_scored"],
            f"{prefix}_avg_goals_conceded":  latest["avg_goals_conceded"],
            f"{prefix}_win_rate":            latest["win_rate"],
            f"{prefix}_draw_rate":           latest["draw_rate"],
            f"{prefix}_avg_points":          latest["avg_points"],
            f"{prefix}_clean_sheet_rate":    latest["clean_sheet_rate"],
            f"{prefix}_form_points":         latest["form_points"],
            f"{prefix}_form_goal_diff":      latest["form_goal_diff"],
            f"{prefix}_form_goals_scored":   latest["form_goals_scored"],
            f"{prefix}_form_goals_conceded": latest["form_goals_conceded"],
            f"{prefix}_win_loss_streak":     latest["win_loss_streak"],
            f"{prefix}_fifa_rank":           latest["fifa_rank"],
            f"{prefix}_fifa_points":         latest["fifa_points"],
            f"{prefix}_ranking_momentum":    latest["ranking_momentum"],
        }

    def get_h2h_stats(self, home_team: str, away_team: str) -> dict:
        """
        Fetch the most recent H2H feature row for this pair from features.csv. Extended to include h2h_avg_goal_diff, h2h_avg_total_goals, and h2h_days_since_last.
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
                "h2h_home_win_rate":   0.5,
                "h2h_away_win_rate":   0.5,
                "h2h_draw_rate":       0.0,
                "h2h_total":           0,
                "h2h_avg_goal_diff":   0.0,
                "h2h_avg_total_goals": 2.5,
                "h2h_days_since_last": 3650,
            }

        candidates = []

        if not direct.empty:
            r = direct.sort_values("date").iloc[-1]
            candidates.append({
                "date":                r["date"],
                "h2h_home_win_rate":   r["h2h_home_win_rate"],
                "h2h_away_win_rate":   r["h2h_away_win_rate"],
                "h2h_draw_rate":       r["h2h_draw_rate"],
                "h2h_total":           r["h2h_total"],
                "h2h_avg_goal_diff":   r["h2h_avg_goal_diff"],
                "h2h_avg_total_goals": r["h2h_avg_total_goals"],
                "h2h_days_since_last": r["h2h_days_since_last"],
            })

        if not reverse.empty:
            r = reverse.sort_values("date").iloc[-1]
            candidates.append({
                "date":                r["date"],
                # Flip home/away perspective
                "h2h_home_win_rate":   r["h2h_away_win_rate"],
                "h2h_away_win_rate":   r["h2h_home_win_rate"],
                "h2h_draw_rate":       r["h2h_draw_rate"],
                "h2h_total":           r["h2h_total"],
                "h2h_avg_goal_diff":  -r["h2h_avg_goal_diff"],
                "h2h_avg_total_goals": r["h2h_avg_total_goals"],
                "h2h_days_since_last": r["h2h_days_since_last"],
            })

        latest = sorted(candidates, key=lambda x: x["date"])[-1]
        latest.pop("date")
        return latest

    def build_match_features(
        self,
        home_team: str,
        away_team: str,
        is_world_cup: int = 1,
        is_neutral: int = 1,
        is_knockout: int = 0,
    ) -> pd.DataFrame:
        """
        Build one ML-ready feature row for a selected match
        """
        if home_team == away_team:
            raise ValueError("Home team and away team cannot be the same.")

        row = {}
        row.update(self.get_latest_team_stats(home_team, "home"))
        row.update(self.get_latest_team_stats(away_team, "away"))
        row.update(self.get_h2h_stats(home_team, away_team))

        row["rank_diff"]   = row["away_fifa_rank"]   - row["home_fifa_rank"]
        row["points_diff"] = row["home_fifa_points"] - row["away_fifa_points"]

        row["is_world_cup"] = int(is_world_cup)
        row["is_neutral"]   = int(is_neutral)
        row["is_knockout"]  = int(is_knockout)

        X = pd.DataFrame([row])[FEATURE_COLS]
        return X

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_world_cup: int = 1,
        is_neutral: int = 1,
        match_stage: str = "group",
    ) -> dict:
        """
        Predict outcome and score for a match.

        match_stage: "group" or "knockout". Controls which draw margin is applied 
        """
        if match_stage not in ("group", "knockout"):
            raise ValueError(
                f"match_stage must be 'group' or 'knockout', got {match_stage!r}"
            )
        draw_margin = (
            DRAW_MARGIN_KNOCKOUT if match_stage == "knockout" else DRAW_MARGIN_GROUP_STAGE
        )

        X = self.build_match_features(
            home_team=home_team,
            away_team=away_team,
            is_world_cup=is_world_cup,
            is_neutral=is_neutral,
            is_knockout=1 if match_stage == "knockout" else 0,
        )

        if hasattr(self.outcome_model, "predict_proba"):
            raw_proba = self.outcome_model.predict_proba(X)
            classes = list(self.outcome_model.classes_)

            probabilities = np.asarray(raw_proba).reshape(-1)

            prob_dict = {}
            for cls, prob in zip(classes, probabilities):
                prob_dict[RESULT_LABELS.get(cls, str(cls))] = float(prob)

            outcome_pred_value = apply_draw_margin(probabilities, classes, margin=draw_margin)
        else:
            prob_dict = {}
            # Fallback if the model has no predict_proba
            outcome_pred = self.outcome_model.predict(X)
            outcome_pred_value = outcome_pred[0] if isinstance(outcome_pred, np.ndarray) else outcome_pred

        raw_score_pred = self.score_model.predict(X)

        score_pred = np.asarray(raw_score_pred).reshape(-1)

        if len(score_pred) < 2:
            raise ValueError(
                f"Score model returned unexpected output: {raw_score_pred}"
            )

        score_pred = np.clip(score_pred, 0, None)

        expected_home_goals = float(score_pred[0])
        expected_away_goals = float(score_pred[1])

        most_likely_outcome = max(prob_dict, key=prob_dict.get)
        scoreline_target = RESULT_LABELS.get(outcome_pred_value, most_likely_outcome)

        if scoreline_target == "Home Win":
            predicted_home_goals = max(1, int(round(expected_home_goals)))
            predicted_away_goals = max(0, int(round(expected_away_goals)))
            
            if predicted_home_goals <= predicted_away_goals:
                predicted_home_goals = predicted_away_goals + 1
                
        elif scoreline_target == "Away Win":
            predicted_home_goals = max(0, int(round(expected_home_goals)))
            predicted_away_goals = max(1, int(round(expected_away_goals)))
            
            if predicted_away_goals <= predicted_home_goals:
                predicted_away_goals = predicted_home_goals + 1
                
        else:
            avg_goals = int(round((expected_home_goals + expected_away_goals) / 2))
            predicted_home_goals = max(0, avg_goals)
            predicted_away_goals = predicted_home_goals

        return {
            "home_team": home_team,
            "away_team": away_team,
            "match_stage": match_stage,
            "draw_margin_used": draw_margin,
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict a FIFA World Cup match outcome and score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single match, knockout stage
  python src/predictor.py "Spain" "Austria" --stage knockout

  # Multiple matches at once
  python src/predictor.py "Spain" "Austria" "Portugal" "Croatia" "Switzerland" "Algeria" --stage knockout

  # Group stage (default)
  python src/predictor.py "Brazil" "France"
        """
    )
    parser.add_argument(
        "teams",
        nargs="+",
        metavar="TEAM",
        help="Team names in pairs: HOME AWAY [HOME AWAY ...]"
             "Must be even number of names"
    )
    parser.add_argument(
        "--stage",
        choices=["group", "knockout"],
        default="group",
        help="Match stage controls draw margin (default is group stage)"
    )
    args = parser.parse_args()

    if len(args.teams) % 2 != 0:
        parser.error("Teams must be provided in pairs")

    predictor = WorldCupPredictor()
    fixtures = list(zip(args.teams[::2], args.teams[1::2]))

    for home, away in fixtures:
        print()
        try:
            r = predictor.predict_match(home, away, match_stage=args.stage)
        except ValueError as e:
            print(f"⚠️  {home} vs {away}: {e}")
            continue

        probs = r["probabilities"]
        stage_label = "Knockout" if args.stage == "knockout" else "Group Stage"

        print(f"{'─' * 50}")
        print(f"  {home}  vs  {away}  [{stage_label}]")
        print(f"{'─' * 50}")
        print(f"  Prediction:  {r['predicted_result']}")
        print(f"  Score:       {home} {r['predicted_home_goals']} – {r['predicted_away_goals']} {away}")
        print(f"  Exp. goals:  {r['expected_home_goals']:.2f} – {r['expected_away_goals']:.2f}")
        print(f"  Probability: {home} {probs.get('Home Win', 0):.1%}  |  "
              f"Draw {probs.get('Draw', 0):.1%}  |  "
              f"{away} {probs.get('Away Win', 0):.1%}")
    print()
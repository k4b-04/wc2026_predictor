"""
FIFA World Cup 2026 Match Outcome Predictor — Streamlit App

Two tabs:
  1. Predict Match   — pick two teams, get outcome + score prediction
  2. Model Insights   — feature importance, dataset overview, model performance
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predictor import WorldCupPredictor

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(
    page_title="WC 2026 Match Predictor",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> WorldCupPredictor:
    """Load the trained models + features once per session"""
    return WorldCupPredictor()


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    path = MODELS_DIR / "feature_importance.csv"
    return pd.read_csv(path)


@st.cache_data
def load_features_summary() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    return pd.read_csv(path, parse_dates=["date"])


def render_probability_chart(probabilities: dict, home_team: str, away_team: str):
    """Horizontal bar chart of Home Win / Draw / Away Win probabilities"""
    labels = ["Home Win", "Draw", "Away Win"]
    display_labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
    values = [probabilities.get(lbl, 0) * 100 for lbl in labels]
    colors = ["#2E7D32", "#9E9E9E", "#C62828"]

    fig = go.Figure(go.Bar(
        x=values,
        y=display_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="Probability (%)",
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_range=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_card(result: dict):
    """Main scoreline + outcome summary"""
    home, away = result["home_team"], result["away_team"]
    h_goals, a_goals = result["predicted_home_goals"], result["predicted_away_goals"]

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"### {home}")
    with col2:
        st.markdown(
            f"<h2 style='text-align:center;'>{h_goals} – {a_goals}</h2>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(f"### {away}")

    st.caption(
        f"Most likely outcome: **{result['most_likely_outcome']}** · "
        f"Expected goals: {result['expected_home_goals']:.2f} – {result['expected_away_goals']:.2f}"
    )


st.title("FIFA World Cup 2026 Match Predictor")
st.markdown(
    "Predicts match winners and scorelines using historical international "
    "results, FIFA rankings, recent form, and head-to-head records"
)

try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(
        "Model files not found. Run `python src/data_preparation.py` "
        "and `python src/model_training.py` first.\n\n"
        f"Details: {e}"
    )
    st.stop()

tab_predict, tab_insights = st.tabs(["Predict Match", "Model Insights"])


with tab_predict:
    teams = predictor.get_teams()

    col_home, col_vs, col_away = st.columns([3, 1, 3])
    with col_home:
        home_team = st.selectbox("Home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col_vs:
        st.markdown("<h3 style='text-align:center; padding-top:1.8rem;'>vs</h3>", unsafe_allow_html=True)
    with col_away:
        away_options = [t for t in teams if t != home_team]
        default_away = "France" if "France" in away_options else away_options[0]
        away_team = st.selectbox("Away team", away_options, index=away_options.index(default_away))

    match_stage = st.radio(
        "Match stage",
        options=["group", "knockout"],
        format_func=lambda s: "Group Stage" if s == "group" else "Knockout Round",
        horizontal=True,
        help=(
            "Controls how readily the model calls a draw. Group stage matches "
            "historically draw more often (~24.7%) than knockout matches in "
            "90-minutes (~21.4%) -- players push harder for a winner when there's "
            "no second leg and elimination is immediate. Knockout matches always "
            "produce a winner on paper via extra time/penalties, but the model "
            "is predicting the 90-minute scoreline, which can still tie."
        ),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        is_world_cup = st.checkbox("World Cup match", value=True)
    with col_b:
        # Knockout matches are conventionally always at a neutral venue
        is_neutral = st.checkbox("Neutral venue", value=(match_stage == "knockout"))

    predict_clicked = st.button("Predict Match", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            result = predictor.predict_match(
                home_team=home_team,
                away_team=away_team,
                is_world_cup=int(is_world_cup),
                is_neutral=int(is_neutral),
                match_stage=match_stage,
            )
        except ValueError as e:
            st.warning(str(e))
        else:
            st.divider()
            render_prediction_card(result)
            st.caption(
                f"Stage: **{'Group Stage' if match_stage == 'group' else 'Knockout Round'}** "
                f"(draw margin = {result['draw_margin_used']})"
            )
            st.divider()
            st.markdown("#### Outcome Probabilities")
            render_probability_chart(result["probabilities"], home_team, away_team)

            with st.expander("View raw feature values used for this prediction"):
                st.dataframe(result["feature_row"].T.rename(columns={0: "value"}))


with tab_insights:
    st.markdown("#### Feature Importance (Outcome Classifier)")
    fi_df = load_feature_importance().sort_values("importance", ascending=True).tail(15)

    fig_fi = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_fi.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    st.markdown("#### Dataset Overview")
    features_df = load_features_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total matches", f"{len(features_df):,}")
    c2.metric("Teams covered", f"{len(set(features_df['home_team']) | set(features_df['away_team']))}")
    c3.metric("Date range", f"{features_df['date'].dt.year.min()}–{features_df['date'].dt.year.max()}")
    c4.metric("World Cup matches", f"{int(features_df['is_world_cup'].sum()):,}")

    st.markdown("#### Historical Outcome Distribution")
    outcome_counts = features_df["result"].map(
        {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    ).value_counts()

    fig_outcomes = px.pie(
        values=outcome_counts.values,
        names=outcome_counts.index,
        color=outcome_counts.index,
        color_discrete_map={"Home Win": "#2E7D32", "Draw": "#9E9E9E", "Away Win": "#C62828"},
        hole=0.4,
    )
    fig_outcomes.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_outcomes, use_container_width=True)
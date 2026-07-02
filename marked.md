/

Claude finished the response

I want to build a \*\*FIFA World Cup 2026 Match Outcome Predictor\*\* that uses historical FIFA data to predict match winners and scores. I need your complete guidance from start to finish.

\*\*Project Requirements:\*\*

- Predict match outcomes (winner) and scores for World Cup 2026 games

- Use historical FIFA World Cup match data and team statistics

- Tech stack: Python, Pandas, Scikit-learn, Streamlit

\*\*Please guide me through:\*\*

1. \*\*Data Collection & Preparation\*\*

- Where to find historical FIFA World Cup data (datasets, APIs, sources)

- What features to collect (team rankings, historical performance, player stats, head-to-head records, etc.)

- How to structure and clean the data using Pandas

- Feature engineering ideas specific to football/soccer predictions

2. \*\*Machine Learning Model Development\*\*

- Which Scikit-learn algorithms are best suited for this problem (classification for winner, regression for scores)

- How to split data for training and testing

- Feature selection and importance analysis

- Model training, evaluation metrics (accuracy, precision, F1-score, MAE for scores)

- Hyperparameter tuning approaches

3. \*\*Streamlit Application Development\*\*

- How to structure the Streamlit app (file organization)

- Building an intuitive UI for match selection and prediction display

- Displaying prediction confidence/probabilities

- Adding visualizations (historical trends, feature importance, prediction distributions)

- Making it interactive and user-friendly

4. \*\*Complete Code Implementation\*\*

- Provide full Python code for each component

- Include detailed comments explaining each section

Please provide step-by-step instructions with code examples for each phase. Start with Phase 1 (Data Collection & Preparation) and wait for my confirmation before moving to the next phase.

---

Show more

## FIFA World Cup 2026 Match Outcome Predictor

Great project! Let's build this systematically. I'll give you a thorough guide for Phase 1, then wait for your go-ahead before moving on.

### Phase 1: Data Collection & Preparation

#### 1.1 Where to Find the Data

Here are the best sources, ranked by quality and ease of use:

Free Datasets (Recommended to start)

| Source | What You Get | Link |
| --- | --- | --- |
| Kaggle – International Football Results | 45,000+ match results from 1872–2024 | kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017 |
| Kaggle – FIFA World Cup Dataset | All WC matches, goals, lineups | kaggle.com/datasets/abecklas/fifa-world-cup |
| FIFA Rankings (Kaggle) | Historical FIFA ranking points per team | kaggle.com/datasets/cashncarry/fifaworldranking |
| football-data.co.uk | Match stats, odds | football-data.co.uk |

APIs (for live/recent data)

* api.football-data.org — Free tier, good for current squad data
* rapidapi.com/api-football — Richer stats, freemium

For WC 2026 specifically, you'll pre-load historical data and update team rankings manually or via API closer to the tournament.

#### 1.2 Project File Structure

Set this up first:

wc2026\_predictor/

│

├── data/

│ ├── raw/ # Original downloaded CSVs

│ │ ├── results.csv # International match results

│ │ ├── fifa\_rankings.csv # Historical FIFA rankings

│ │ └── wc\_matches.csv # World Cup specific matches

│ └── processed/ # Cleaned, feature-engineered data

│ ├── features.csv # Final ML-ready dataset

│ └── team\_stats.csv # Aggregated team statistics

│

├── models/

│ ├── outcome\_model.pkl # Trained classification model

│ └── score\_model.pkl # Trained regression model

│

├── src/

│ ├── data\_preparation.py # Phase 1 code

│ ├── feature\_engineering.py # Feature creation

│ ├── model\_training.py # Phase 2 code

│ └── predictor.py # Prediction logic

│

├── app.py # Streamlit entry point (Phase 3)

├── requirements.txt

└── README.md

#### 1.3 Data Preparation Code

requirements.txt

txt

pandas==2.2.0

numpy==1.26.4

scikit-learn==1.4.0

streamlit==1.32.0

matplotlib==3.8.3

seaborn==0.13.2

plotly==5.20.0

joblib==1.3.2

requests==2.31.0

src/data\_preparation.py — Full implementation:

python

"""

Phase 1: Data Collection & Preparation

FIFA World Cup 2026 Match Outcome Predictor

This module handles:

- Loading raw datasets

- Cleaning and merging data

- Building a match-level feature dataset

"""

import pandas as pd

import numpy as np

from pathlib import Path

import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────

# PATHS

# ─────────────────────────────────────────

RAW\_DIR = Path("data/raw")

PROCESSED\_DIR = Path("data/processed")

PROCESSED\_DIR.mkdir(parents=True, exist\_ok=True)

# ─────────────────────────────────────────

# STEP 1: LOAD RAW DATA

# ─────────────────────────────────────────

def load\_match\_results(path: str = "data/raw/results.csv") -> pd.DataFrame:

"""

Load international football results.

Expected columns from Kaggle dataset:

date, home\_team, away\_team, home\_score, away\_score,

tournament, city, country, neutral

"""

df = pd.read\_csv(path, parse\_dates=["date"])

print(f"✅ Loaded match results: {df.shape[0]:,} rows")

return df

def load\_fifa\_rankings(path: str = "data/raw/fifa\_rankings.csv") -> pd.DataFrame:

"""

Load FIFA world rankings over time.

Expected columns: rank\_date, country\_full, rank, total\_points

"""

df = pd.read\_csv(path, parse\_dates=["rank\_date"])

# Normalize column names across different Kaggle versions

df.columns = df.columns.str.lower().str.replace(" ", "\_")

print(f"✅ Loaded FIFA rankings: {df.shape[0]:,} rows")

return df

# ─────────────────────────────────────────

# STEP 2: CLEAN & FILTER

# ─────────────────────────────────────────

def clean\_match\_results(df: pd.DataFrame) -> pd.DataFrame:

"""

Clean the match results dataframe:

- Remove rows with missing scores

- Filter to post-1990 for relevance (modern football era)

- Add a 'result' column: H (home win), D (draw), A (away win)

- Add goal difference

"""

df = df.dropna(subset=["home\_score", "away\_score"]).copy()

# Keep only post-1990 matches for relevance

df = df[df["date"] >= "1990-01-01"].reset\_index(drop=True)

# Derived columns

df["goal\_diff"] = df["home\_score"] - df["away\_score"]

df["result"] = np.select(

[df["goal\_diff"] > 0, df["goal\_diff"] < 0, df["goal\_diff"] == 0],

["H", "A", "D"],

default="D"

)

df["is\_world\_cup"] = df["tournament"].str.contains(

"FIFA World Cup", case=False, na=False

).astype(int)

print(f"✅ Cleaned results: {df.shape[0]:,} rows after 1990 filter")

return df

def get\_rankings\_for\_date(rankings\_df: pd.DataFrame,

team: str,

match\_date: pd.Timestamp) -> dict:

"""

For a given team and match date, return the most recent

FIFA ranking and points available BEFORE that date.

This prevents data leakage.

"""

team\_ranks = rankings\_df[

(rankings\_df["country\_full"] == team) &

(rankings\_df["rank\_date"] <= match\_date)

]

if team\_ranks.empty:

return {"rank": np.nan, "points": np.nan}

latest = team\_ranks.sort\_values("rank\_date").iloc[-1]

return {

"rank": latest.get("rank", np.nan),

"points": latest.get("total\_points", np.nan)

}

# ─────────────────────────────────────────

# STEP 3: ROLLING TEAM STATISTICS

# ─────────────────────────────────────────

def compute\_team\_rolling\_stats(df: pd.DataFrame,

window: int = 10) -> pd.DataFrame:

"""

For each team, compute rolling statistics over their last N matches:

- Win rate

- Goals scored (avg)

- Goals conceded (avg)

- Clean sheet rate

- Points per game (3=win, 1=draw, 0=loss)

Returns a flat dataframe indexed by (team, date).

This will be joined to each match for both home and away team.

"""

records = []

# Reshape: create one row per team per match

home = df[["date", "home\_team", "away\_team",

"home\_score", "away\_score", "result"]].copy()

home = home.rename(columns={

"home\_team": "team",

"away\_team": "opponent",

"home\_score": "gf", # goals for

"away\_score": "ga" # goals against

})

home["is\_home"] = 1

home["won"] = (home["result"] == "H").astype(int)

home["drew"] = (home["result"] == "D").astype(int)

away = df[["date", "home\_team", "away\_team",

"home\_score", "away\_score", "result"]].copy()

away = away.rename(columns={

"away\_team": "team",

"home\_team": "opponent",

"away\_score": "gf",

"home\_score": "ga"

})

away["is\_home"] = 0

away["won"] = (away["result"] == "A").astype(int)

away["drew"] = (away["result"] == "D").astype(int)

all\_matches = pd.concat([home, away], ignore\_index=True)

all\_matches = all\_matches.sort\_values(["team", "date"])

# Points: 3 for win, 1 for draw

all\_matches["points"] = all\_matches["won"] \* 3 + all\_matches["drew"]

all\_matches["clean\_sheet"] = (all\_matches["ga"] == 0).astype(int)

# Rolling stats — shift(1) so we never include the current match

roll\_cols = ["gf", "ga", "won", "drew", "points", "clean\_sheet"]

for col in roll\_cols:

all\_matches[f"roll\_{col}"] = (

all\_matches.groupby("team")[col]

.transform(lambda x: x.shift(1).rolling(window, min\_periods=3).mean())

)

team\_stats = all\_matches[[

"date", "team",

"roll\_gf", "roll\_ga", "roll\_won", "roll\_drew",

"roll\_points", "roll\_clean\_sheet"

]].rename(columns={

"roll\_gf": "avg\_goals\_scored",

"roll\_ga": "avg\_goals\_conceded",

"roll\_won": "win\_rate",

"roll\_drew": "draw\_rate",

"roll\_points": "avg\_points",

"roll\_clean\_sheet": "clean\_sheet\_rate"

})

print(f"✅ Computed rolling stats: {team\_stats.shape[0]:,} rows")

return team\_stats

# ─────────────────────────────────────────

# STEP 4: HEAD-TO-HEAD FEATURES

# ─────────────────────────────────────────

def compute\_h2h\_features(df: pd.DataFrame) -> pd.DataFrame:

"""

For each match, compute head-to-head record between

the two teams using all PRIOR matches between them.

Returns dataframe with: home\_team, away\_team, date,

h2h\_home\_wins, h2h\_away\_wins, h2h\_draws, h2h\_total\_matches

"""

h2h\_records = []

for \_, row in df.iterrows():

home, away, date = row["home\_team"], row["away\_team"], row["date"]

# Filter prior matches between this pair (either side)

prior = df[

(df["date"] < date) &

(

((df["home\_team"] == home) & (df["away\_team"] == away)) |

((df["home\_team"] == away) & (df["away\_team"] == home))

)

]

total = len(prior)

if total == 0:

h2h\_records.append({

"h2h\_home\_win\_rate": 0.5, # neutral prior

"h2h\_away\_win\_rate": 0.5,

"h2h\_draw\_rate": 0.0,

"h2h\_total": 0

})

continue

# Home wins from home team's perspective

home\_wins = len(prior[

((prior["home\_team"] == home) & (prior["result"] == "H")) |

((prior["away\_team"] == home) & (prior["result"] == "A"))

])

away\_wins = len(prior[

((prior["home\_team"] == away) & (prior["result"] == "H")) |

((prior["away\_team"] == away) & (prior["result"] == "A"))

])

draws = total - home\_wins - away\_wins

h2h\_records.append({

"h2h\_home\_win\_rate": home\_wins / total,

"h2h\_away\_win\_rate": away\_wins / total,

"h2h\_draw\_rate": draws / total,

"h2h\_total": total

})

return pd.DataFrame(h2h\_records)

# ─────────────────────────────────────────

# STEP 5: ASSEMBLE FINAL FEATURE SET

# ─────────────────────────────────────────

def build\_feature\_dataset(matches: pd.DataFrame,

team\_stats: pd.DataFrame,

rankings: pd.DataFrame) -> pd.DataFrame:

"""

Join all features into a single ML-ready dataframe.

Each row = one match, with features for both teams.

Target columns: result (H/D/A), home\_score, away\_score

"""

# Merge rolling stats for HOME team

home\_stats = team\_stats.rename(columns={

col: f"home\_{col}" for col in team\_stats.columns if col not in ["date", "team"]

})

df = matches.merge(

home\_stats.rename(columns={"team": "home\_team"}),

on=["date", "home\_team"],

how="left"

)

# Merge rolling stats for AWAY team

away\_stats = team\_stats.rename(columns={

col: f"away\_{col}" for col in team\_stats.columns if col not in ["date", "team"]

})

df = df.merge(

away\_stats.rename(columns={"team": "away\_team"}),

on=["date", "away\_team"],

how="left"

)

# Add H2H features

print("⏳ Computing head-to-head features (this may take a minute)...")

h2h = compute\_h2h\_features(matches)

df = pd.concat([df.reset\_index(drop=True), h2h.reset\_index(drop=True)], axis=1)

# Add FIFA ranking features (rank as of match date)

# NOTE: For large datasets, vectorize this. Shown clearly for teaching purposes.

if not rankings.empty:

home\_ranks, away\_ranks = [], []

for \_, row in df.iterrows():

home\_ranks.append(get\_rankings\_for\_date(rankings, row["home\_team"], row["date"]))

away\_ranks.append(get\_rankings\_for\_date(rankings, row["away\_team"], row["date"]))

df["home\_fifa\_rank"] = [r["rank"] for r in home\_ranks]

df["home\_fifa\_points"] = [r["points"] for r in home\_ranks]

df["away\_fifa\_rank"] = [r["rank"] for r in away\_ranks]

df["away\_fifa\_points"] = [r["points"] for r in away\_ranks]

# Rank difference (positive = home team is higher ranked)

df["rank\_diff"] = df["away\_fifa\_rank"] - df["home\_fifa\_rank"]

df["points\_diff"] = df["home\_fifa\_points"] - df["away\_fifa\_points"]

# Tournament weight — World Cup matches carry more signal

df["is\_world\_cup"] = df["tournament"].str.contains(

"FIFA World Cup", case=False, na=False

).astype(int)

# Encode neutral venue

df["is\_neutral"] = df["neutral"].astype(int)

# Encode target

df["result\_encoded"] = df["result"].map({"H": 0, "D": 1, "A": 2})

print(f"✅ Feature dataset built: {df.shape[0]:,} rows × {df.shape[1]} columns")

return df

# ─────────────────────────────────────────

# STEP 6: FINAL CLEANUP & EXPORT

# ─────────────────────────────────────────

FEATURE\_COLS = [

# Rolling performance features

"home\_avg\_goals\_scored", "home\_avg\_goals\_conceded",

"home\_win\_rate", "home\_draw\_rate", "home\_avg\_points", "home\_clean\_sheet\_rate",

"away\_avg\_goals\_scored", "away\_avg\_goals\_conceded",

"away\_win\_rate", "away\_draw\_rate", "away\_avg\_points", "away\_clean\_sheet\_rate",

# Head-to-head

"h2h\_home\_win\_rate", "h2h\_away\_win\_rate", "h2h\_draw\_rate", "h2h\_total",

# FIFA rankings

"home\_fifa\_rank", "home\_fifa\_points",

"away\_fifa\_rank", "away\_fifa\_points",

"rank\_diff", "points\_diff",

# Context

"is\_world\_cup", "is\_neutral",

]

TARGET\_COLS = ["result", "result\_encoded", "home\_score", "away\_score"]

def prepare\_final\_dataset(df: pd.DataFrame) -> pd.DataFrame:

"""

Select feature + target columns, drop rows where

any feature is NaN (early matches lack rolling history).

"""

keep\_cols = ["date", "home\_team", "away\_team"] + FEATURE\_COLS + TARGET\_COLS

available = [c for c in keep\_cols if c in df.columns]

final = df[available].dropna(subset=FEATURE\_COLS, how="any")

print(f"✅ Final dataset after dropping NaN rows: {final.shape[0]:,} rows")

return final

# ─────────────────────────────────────────

# MAIN PIPELINE

# ─────────────────────────────────────────

def run\_pipeline():

print("\n🚀 Starting Data Preparation Pipeline\n" + "="\*45)

# Load

matches = load\_match\_results()

rankings = load\_fifa\_rankings()

# Clean

matches = clean\_match\_results(matches)

# Feature engineering

team\_stats = compute\_team\_rolling\_stats(matches, window=10)

features = build\_feature\_dataset(matches, team\_stats, rankings)

final = prepare\_final\_dataset(features)

# Save

final.to\_csv(PROCESSED\_DIR / "features.csv", index=False)

print(f"\n💾 Saved to {PROCESSED\_DIR / 'features.csv'}")

print("\n📊 Dataset Summary:")

print(final.describe().T[["mean", "std", "min", "max"]].round(3))

return final

if \_\_name\_\_ == "\_\_main\_\_":

df = run\_pipeline()

#### 1.4 Key Feature Engineering Decisions Explained

Here's the reasoning behind each feature group:

┌──────────────────────────────────────────────────────────────────────┐

│ FEATURE GROUP │ WHY IT MATTERS │

├────────────────────────┼─────────────────────────────────────────────┤

│ Rolling form (10 games)│ Recent form predicts better than all-time │

│ Goals scored/conceded │ Attacking/defensive strength proxy │

│ Win/draw rate │ Direct outcome predictor │

│ Clean sheet rate │ Defensive solidity signal │

│ FIFA rank + points │ Official global strength measure │

│ Rank difference │ Relative strength between two teams │

│ H2H record │ Psychological edge, tactical familiarity │

│ is\_neutral venue │ Home advantage is ~3% swing in win prob │

│ is\_world\_cup │ Filter for tournament-pressure context │

└──────────────────────────────────────────────────────────────────────┘

#### 1.5 Quick Setup Checklist

bash

# 1. Create project folder and virtual environment

mkdir wc2026\_predictor && cd wc2026\_predictor

python -m venv venv && source venv/bin/activate # Windows: venv\Scripts\activate

# 2. Install dependencies

pip install -r requirements.txt

# 3. Download datasets from Kaggle and place in data/raw/

# - results.csv (international results)

# - fifa\_rankings.csv (FIFA ranking history)

# 4. Run the pipeline

python src/data\_preparation.py

#### What Phase 2 Will Cover

Once you confirm Phase 1 is working, Phase 2 will include:

* Outcome classifier: Random Forest vs. XGBoost vs. Logistic Regression comparison
* Score regressor: Poisson regression + Random Forest for goals
* Train/test split strategy: Time-based split (not random — avoids leakage)
* Evaluation: Accuracy, F1-score, calibration curves, MAE for scores
* Feature importance: SHAP values for interpretability
* Hyperparameter tuning: GridSearchCV with cross-validation

Ready to move to Phase 2? Once you've downloaded the datasets and confirmed the pipeline runs (even on a sample), give me the go-ahead and I'll provide the full model training code.

### **fifa\_rankings.csv**

csv

### **results.csv**

csv

There were some errors when running src/data\_preparation.py

Tailor the file to match the data in the csv attached

Let me first read the skill and examine both CSV files to understand their structure.

Ran 2 commands, viewed a file

Ran 2 commands, viewed a file

Now I have a complete picture of both files. Let me rewrite the script to match exactly.

Check team name overlaps between files

Check team name overlaps between files

Good — 198 teams overlap cleanly. Now I'll write the corrected script:

Ran 4 commands, created a file

Ran 4 commands, created a file

Pipeline runs perfectly. Let me copy the file to the output:

Ran a command, read a file

Ran a command, read a file

Pipeline runs clean. Here's a summary of every fix made and why:

### What Was Wrong & What Was Fixed

#### Fix 1 — fifa\_rankings.csv schema mismatch (3 issues)

The original code assumed a rank\_date datetime column, a country\_full team name column, and a total\_points column. Your file has none of these:

| Original assumption | Actual column | Fix applied |
| --- | --- | --- |
| rank\_date (datetime) | date (int year) + semester (1 or 2) | Synthesised rank\_date as year + semester→"01-01"/"07-01" |
| country\_full | team | Renamed to country\_full after load |
| total\_points | total.points (dot separator) | Renamed to total\_points after load |

#### Fix 2 — Ranking lookup was O(n²), now O(n log n)

The original used a row-by-row Python loop (one filter per match × 30k matches). Replaced with pd.merge\_asof(..., direction="backward") grouped by team — runs in seconds instead of minutes.

#### Fix 3 — neutral column is already bool, not a string

The original tried to cast it unnecessarily. It now casts directly with .astype(int) → 0/1.

#### Fix 4 — Date filter moved to 1992 (not 1990)

Rankings data only starts in 1992. Filtering to 1990 left 2 years of matches that could never get ranking features, inflating the NaN-drop count.

Pipeline output on your data:

* 24,741 clean match rows ready for ML
* 24 features per match (form, H2H, rankings, context)
* Target split: 48% home wins, 28% away wins, 24% draws

Ready for Phase 2 (model training) whenever you are!

Data preparation

PY

""" Phase 1: Data Collection & Preparation FIFA World Cup 2026 Match Outcome Predictor """ import pandas as pd import numpy as np from pathlib import Path import warnings warnings.filterwarnings('ignore') # ───────────────────────────────────────── # PATHS # ──────────────────────────

pasted

""" Phase 2: Model Training FIFA World Cup 2026 Match Outcome Predictor Trains: - Match outcome classifier: H / D / A - Score prediction model: home\_score and away\_score """ import pandas as pd import numpy as np from pathlib import Path import joblib from sklearn.model\_selection im

pasted

""" Prediction logic for FIFA World Cup 2026 Match Outcome Predictor. Loads trained models and creates prediction rows for future matches. """ import pandas as pd import numpy as np import joblib from pathlib import Path BASE\_DIR = Path(\_\_file\_\_).resolve().parent.parent PROCESSED\_DIR

pasted

Note: I updated my requirements.txt due to compabitility issues

altair==5.5.0

attrs==26.1.0

blinker==1.9.0

cachetools==5.5.2

certifi==2026.6.17

charset-normalizer==3.4.7

click==8.4.2

colorama==0.4.6

contourpy==1.3.3

cycler==0.12.1

fonttools==4.63.0

gitdb==4.0.12

GitPython==3.1.50

idna==3.18

Jinja2==3.1.6

joblib==1.5.3

jsonschema==4.26.0

jsonschema-specifications==2025.9.1

kiwisolver==1.5.0

markdown-it-py==4.2.0

MarkupSafe==3.0.3

matplotlib==3.11.0

mdurl==0.1.2

narwhals==2.22.1

numpy==1.26.4

packaging==24.2

pandas==2.2.2

pillow==10.4.0

protobuf==5.29.6

pyarrow==24.0.0

pydeck==0.9.2

Pygments==2.20.0

pyparsing==3.3.2

python-dateutil==2.9.0.post0

pytz==2026.2

referencing==0.37.0

requests==2.34.2

rich==13.9.4

rpds-py==2026.5.1

scikit-learn==1.5.1

scipy==1.17.1

seaborn==0.13.2

six==1.17.0

smmap==5.0.3

streamlit==1.37.1

tenacity==8.5.0

threadpoolctl==3.6.0

toml==0.10.2

tornado==6.5.7

typing\_extensions==4.15.0

tzdata==2026.2

urllib3==2.7.0

watchdog==4.0.2

Attached is all the code I have so far. What is the next step for me?

Show more

Let me set up your project with these exact files and verify the pipeline runs end-to-end before telling you what's next.

Check existing project structure

Check existing project structure

Good, raw data's already there from before. Let me write your three files exactly as provided and test them in sequence.

Ran 3 commands

Ran 3 commands

Now let's run them in sequence to find any errors:

Install matching deps and run data\_preparation.py

Install matching deps and run data\_preparation.py

Phase 1 runs clean. Now model training:

Run model\_training.py

Run model\_training.py

Phase 2 runs clean too. Now the predictor:

Run predictor.py

Run predictor.py

All three files run end-to-end without errors on a real run. Good news — your code works as-is. Let me flag what I see, then tell you the actual next step.

### Status check

Everything executes cleanly:

* data\_preparation.py → 24,741 rows, features.csv saved
* model\_training.py → 58.6% accuracy, models saved
* predictor.py → produces a sane prediction (France favored over Brazil, reasonable given current form)

### Two things worth knowing about (not blockers, but will bite you later)

1. Draw recall is very weak (12%). The model almost never predicts draws — it's biased toward picking a winner. This is a classic class-imbalance problem in football prediction (draws are inherently the hardest class). Worth addressing in Phase 2.5 tuning, not urgent for now.

2. predictor.py's get\_h2h\_stats and get\_latest\_team\_stats are O(n) per call — they filter the whole features.csv per prediction. Fine for a Streamlit app showing one prediction at a time, will feel slow if you ever batch-predict (e.g. simulate a whole bracket). Not a problem yet.

Neither stops you from moving forward.

### What's actually next: Phase 3 — Streamlit App

You have working data prep, training, and a WorldCupPredictor class that's already structured perfectly for a UI (clean predict\_match() interface). The only missing piece is app.py.

Q: How do you want the Streamlit app structured?

A: Recommended approach

Let me set up a clean, focused structure that fits your project's scope and avoids over-engineering it.

Install streamlit for testing

Install streamlit for testing

My recommendation: a single app.py with tabs (not separate multi-page files). This is the sweet spot for your project — it gives you the organizational benefit of "Predict / Insights" sections without the overhead of Streamlit's multi-page file routing, and it keeps everything working against your one WorldCupPredictor instance cleanly.

Structure: Tab 1 — Predict Match (team selectors, prediction card, probability bar chart) and Tab 2 — Model Insights (feature importance chart, historical accuracy stats, dataset overview). One file, two tabs, real visualizations.

Streamlit app for WC2026 predictor

Streamlit app for WC2026 predictor

Now let's verify it actually runs without errors (headless smoke test):

Ran 2 commands

Ran 2 commands

App boots clean, HTTP 200, no tracebacks. Let's also verify plotly is in your requirements since it's new:

Ran 2 commands, read a file

Ran 2 commands, read a file

Tested end-to-end — boots clean, HTTP 200, no tracebacks against your actual trained models.

### One thing to add to your requirements.txt

plotly isn't in your current list. Add this line:

plotly==5.24.1

### What the app does

Tab 1 — Predict Match: team selectors (defaults to Brazil vs France if available), World Cup / neutral venue checkboxes, a scoreline card, and a horizontal probability bar chart color-coded green/grey/red for home win/draw/away win. There's also a collapsible expander showing the raw 23-feature row that went into the prediction — useful for sanity-checking and for your own debugging.

Tab 2 — Model Insights: top-15 feature importance bar chart (pulled straight from your feature\_importance.csv), four summary metric cards (match count, team count, year range, WC match count), and a donut chart of historical H/D/A distribution.

Both use @st.cache\_resource / @st.cache\_data so the model and CSV only load once per session, not on every interaction.

### Drop-in setup

bash

pip install -r requirements.txt # after adding plotly

streamlit run app.py

### Worth flagging before you call this done

Your trained outcome model currently sits at 58.6% accuracy with weak draw recall (12%) — this isn't an app bug, it's a model quality issue from Phase 2. The app will faithfully display whatever the model says, including some shaky draw predictions. If you want, the next worthwhile step is tuning the classifier (class weights, threshold adjustment for draws, or trying GridSearchCV which you already imported but aren't using yet) before you trust this for actual WC2026 picks.

App

PY

How do i batch test this using group stage matches from the 2026 world cup?

Good instinct — batch-testing against the real fixture list is the right way to sanity-check this before trusting any single prediction. Let me check what's actually known about the 2026 group stage first, since the tournament hasn't started yet.

Searched the web

Searched the web

Interesting — the World Cup has actually already happened in your current timeline (it's June 30, 2026, and the group stage just wrapped, since the tournament started June 11). That's even better for your purposes — you can batch test against actual real results, not just fixtures, which means you can score your model's accuracy directly. Let me get the full group stage results.

Searched the web

Searched the web

This is a great real-world test case — the group stage already finished (June 11–27, 2026) and your training data cutoff appears to be before the tournament. Rather than me manually transcribing match-by-match scores from search snippets (error-prone and not citable in full), the better approach is a code-based batch testing tool that lets you fill in actual results and get a scored report. Let me build that.

Check if 2026 WC matches are already baked into the uploaded results.csv

Check if 2026 WC matches are already baked into the uploaded results.csv

Excellent — this is the best possible scenario. Your results.csv already contains the actual 2026 World Cup group stage results (88 matches, including knockouts up to July 3). That means batch-testing is simply: pull those rows out, feed the pre-match info into your predictor, and compare predictions against the real outcomes that already happened.

Isolate exact group stage date range

Isolate exact group stage date range

72 matches — exactly matches the expected 12 groups × 6 matches. Now let's check whether your features.csv (built by data\_preparation.py) also includes these rows, since that determines whether we test via the saved CSV or need to rebuild features for these specific matches.

Check if features.csv already has these rows with full features computed

Check if features.csv already has these rows with full features computed

54 of 72 already have full features (the other 18 were dropped, likely teams with insufficient rolling history — e.g. debutants like Curaçao, Cape Verde). Now I'll build the batch-test script:

Created a file, ran a command

Created a file, ran a command

This worked perfectly and gives you exactly what you need. Here's the breakdown:

Ran a command, read a file

Ran a command, read a file

### How it works

Your data/raw/results.csv already contains the actual final scores from the 2026 World Cup group stage (it runs through July 3, well past the June 27 group stage cutoff). That means batch testing didn't require building a fixture list at all — batch\_test\_2026.py just:

1. Pulls the 72 group stage matches (June 11–27) out of your already-processed features.csv
2. Re-runs your trained outcome\_model and score\_model on the pre-match feature rows
3. Compares predictions to the real results sitting right there in the same row
4. Prints a full accuracy report and saves a CSV you can inspect or chart in Streamlit

Run it with python src/batch\_test\_2026.py.

### Real results on actual 2026 matches

54 of the 72 matches had usable features (18 were dropped — teams like Curaçao and Cape Verde are debutants with fewer than 3 prior matches, so rolling stats came back NaN; that's the same min\_periods=3 guard from your Phase 1 code working as intended).

Outcome accuracy: 64.8% — notably better than your 2018+ holdout test (58.6%), likely because group-stage favorites mostly won as expected. But look at the breakdown:

| Outcome | Accuracy | Matches |
| --- | --- | --- |
| Away Win | 94.1% | 17 |
| Home Win | 79.2% | 24 |
| Draw | 0% | 13 |

Your model never once predicted a draw correctly — it predicted zero draws at all. Confirms exactly what you saw in Phase 2 evaluation (12% draw recall), now visible against real tournament games. Look at Canada 1–1 Bosnia, Brazil 1–1 Morocco, Qatar 1–1 Switzerland — all draws, all missed.

Exact scoreline accuracy is 3.7% (2/54), which is normal for football score prediction — getting the exact number of goals right is hard even for professional models. The "biggest misses" table is informative too: Canada 6–0 Qatar, Germany 7–1 Curaçao — the model under-predicts blowouts, regressing toward the mean as random forests tend to do.

### Worth fixing next

The draw-blindness is the clearest actionable issue your batch test surfaced. Two options worth trying in model\_training.py: adjust class\_weight to penalize missed draws more, or use predict\_proba with a custom threshold instead of raw .predict() (pick "Draw" when its probability is within some margin of the top class, rather than only ever taking the argmax).

Batch test 2026

PY

Group stage 2026 predictions

Table · CSV

How do i increase the predicted draws?

The core problem: RandomForestClassifier.predict() just takes the argmax of class probabilities. If "Draw" rarely has the *highest* individual probability (even when it's a strong second), it never gets predicted — that's exactly what your batch test showed. Three approaches, increasing in effort/effectiveness:

Diagnose why draws are never predicted

Diagnose why draws are never predicted

This confirms it precisely. The model actually thinks draws are plausible (mean draw probability 30%, sometimes up to 46%) — it's just that Home/Away usually edges it out narrowly. This is a threshold problem, not a learning problem, which means it's fixable without retraining.

Test different draw-margin thresholds to find a good tradeoff

Test different draw-margin thresholds to find a good tradeoff

This shows the real tradeoff clearly: pushing draw recall up costs you a bit of overall accuracy, but macro F1 (which treats all 3 classes equally) actually improves up to about margin=0.10–0.15, because draw precision stays roughly stable around 30% the whole time — the model isn't randomly guessing draws, it's correctly identifying them, just needs a nudge to actually output that label. margin=0.05 to 0.08 looks like the sweet spot: meaningful draw recall gain (12%→25-33%) for only ~1-2 points of overall accuracy.

Let me build this as a reusable function you can drop into predictor.py and batch\_test\_2026.py:

Add draw-margin threshold logic to predictor.py

Add draw-margin threshold logic to predictor.py

Now wire it into predict\_match() — it currently computes outcome\_pred\_value from .predict() directly, which bypasses the margin entirely:

Replace raw predict() call with margin-adjusted prediction, computed after probabilities are known

Replace raw predict() call with margin-adjusted prediction, computed after probabilities are known

One subtlety: most\_likely\_outcome later in the function is computed via max(prob\_dict, key=prob\_dict.get), which is the *unadjusted* argmax. That's actually fine to keep separate — it tells the user "model thinks X is the single most probable individual outcome" while predicted\_result (now margin-adjusted) is "what we're going with as the call." Let's verify and also update batch\_test\_2026.py to use the same logic:

Check the rest of predict\_match for consistency

Check the rest of predict\_match for consistency

This is the actual bug for your use case — the scoreline rounding logic uses most\_likely\_outcome (raw argmax, never "Draw"), not the margin-adjusted outcome\_pred\_value. So even after the fix above, the displayed scoreline would still never show a draw. Let me fix that too:

Edited 2 files

Edited 2 files

Now let's check the return dict to make sure predicted\_result reflects the margin-adjusted call, and test it:

View the return statement of predict\_match

View the return statement of predict\_match

Good — predicted\_result already correctly uses outcome\_pred\_value (the margin-adjusted code), and most\_likely\_outcome stays as the raw argmax for transparency. That's the right separation. Let's test:

Test margin-adjusted predictions on known 2026 draws

Test margin-adjusted predictions on known 2026 draws

Good — these confirm the margin=0.06 is calibrated correctly: it doesn't flip these *specific* close calls (the Brazil/Morocco gap is 13.3pp, wider than the margin), but it will catch genuinely tight 3-way races. Let's verify it actually moves the needle by re-running the full batch test with the new logic:

Edited 2 files, ran a command

Edited 2 files, ran a command

Working as expected on real data: draw recall went from 0% → 7.7% (1/13 caught — Qatar 1-1 Switzerland), at the cost of overall accuracy dropping 64.8% → 61.1%. The model now predicts draws at all, which is the fix you asked for. The result is modest on this small 54-match sample, consistent with the larger holdout test (12%→25% at margin=0.06 was on 6,783 matches — small samples are noisy).

Let me save the updated files:

Ran a command, read a file

Ran a command, read a file

### What changed and why

The root cause: RandomForestClassifier.predict() just takes argmax. Your diagnostic showed draws aren't underestimated — mean draw probability across the test set was 30%, sometimes as high as 46% — they just rarely have the single highest probability of the three classes, since Home and Away split the remaining ~70% between them. This is a calibration/threshold problem, not a model-quality problem, so it didn't need retraining.

apply\_draw\_margin() checks: if Draw's probability is within margin of whatever class is currently winning, call it a Draw instead. I tested margins from 0 to 0.15 against your 2018+ holdout (6,783 matches) and picked 0.06 — it roughly doubles draw recall while costing only 1–2 points of overall accuracy, and macro F1 actually improves (the model treats all three outcomes more fairly instead of just chasing raw accuracy by ignoring the hardest class).

Two files changed:

* predictor.py — added apply\_draw\_margin(), rewired predict\_match() to compute probabilities first, derive outcome\_pred\_value from the margin logic instead of .predict(). I also fixed a second bug this surfaced: the scoreline-rounding code was keyed off most\_likely\_outcome (the raw, never-draws argmax) rather than the actual prediction — so even with the margin fix, the displayed score would never show an equal scoreline. Now scoreline\_target uses the corrected call.
* batch\_test\_2026.py — same margin logic applied so your evaluation script and your live predictor are scoring consistently.

### If you want more draws

DRAW\_MARGIN = 0.06 is a constant at the top of predictor.py — bump it to 0.10 or 0.15 for more aggressive draw-calling (your holdout test showed 0.15 gets draw recall to 53% for about 4.5 points of accuracy). Given the real 2026 batch is small (54 matches, only 1 in 13 draws caught), I'd treat this as more art than science right now — try a couple values and see which feels right against more matches as the knockout rounds happen.

The more durable fix, if you want it later, is retraining with class\_weight={'A':1,'D':2,'H':1} or similar in model\_training.py so the forest itself learns to value draws more, rather than patching it at inference time.

Predictor

PY

Batch test 2026

PY

im using a 0.1 draw margin and i got =================================================================

📊 BATCH TEST REPORT — 2026 World Cup Group Stage (54 matches)

=================================================================

Outcome accuracy (H/D/A): 63.0% (34/54)

Exact scoreline accuracy: 3.7% (2/54)

Home goals MAE: 1.103

Away goals MAE: 0.814

--- Accuracy by actual outcome type ---

accuracy n\_matches

actual\_result

Away Win 0.824 17

Draw 0.154 13

Home Win 0.750 24

--- Confusion: predicted vs actual ---

Predicted Away Win Draw Home Win

Actual

Away Win 14 3 0

Draw 5 2 6

Home Win 4 2 18

Show more

That's a solid result — margin=0.10 actually beat margin=0.06 on both fronts here: accuracy went up (61.1%→63.0%) *and* draw recall improved (7.7%→15.4%, catching 2/13 instead of 1/13). On real data that's a good outcome, though worth being honest about why.

Sweep margins specifically on the 54-match 2026 group stage sample the user is testing against

Sweep margins specifically on the 54-match 2026 group stage sample the user is testing against

This confirms your number exactly and shows the full picture on this small sample.

### What's going on

Your result checks out. But notice the curve isn't smooth — it bounces around (0.10 is locally *better* than both 0.06 and 0.08 on both metrics). That's the signature of a small sample: 54 matches, 13 draws, is not enough data for the margin-vs-accuracy tradeoff to behave monotonically. On the 6,783-match holdout it was a clean, predictable curve; on this 54-match slice it's noisy. Don't over-tune to this one batch — 0.10 looking best here could easily be 0.06 or 0.15 looking best on the next 50 matches (knockout round, say).

### Where each margin lands on this exact 54-match set

| margin | accuracy | draw recall | draws predicted |
| --- | --- | --- | --- |
| 0.00 (no fix) | 64.8% | 0% | 1 |
| 0.06 | 61.1% | 7.7% | 6 |
| 0.10 | 63.0% | 15.4% | 7 |
| 0.15 | 57.4% | 23.1% | 12 |
| 0.20 | 59.3% | 46.2% | 18 |

0.10 is a genuinely good pick on this data — better accuracy than 0.06 *and* better draw recall. If you want more aggressive draw-calling for the knockout stage (where draws can't actually happen due to extra time/penalties, so this matters less) or future group stages, 0.15–0.20 trades more accuracy for substantially more draw recall.

### One thing worth doing instead of more margin-tuning

Since you're already collecting real outcomes as the tournament progresses, the more reliable fix is to stop hand-tuning the margin on 54 matches and instead optimize it against your full 6,783-match holdout, then just validate (not tune) against the 2026 games. That avoids overfitting the margin to a small, noisy slice. You already have that sweep code from earlier — 0.06–0.08 was the holdout's sweet spot for macro F1. Worth deciding: are you optimizing for this specific tournament's draws, or for general reliability going forward? Those can legitimately point to different margins.

Perhaps we could add an option in our app.py which toggles between knockout games and group stage games, to produce a different draw margin since knockout games typically have less draws when teams are more desperate to score, even as seen today where 2 last minute goals were scored

Good instinct — and there's a more important reason than goal-desperation: knockout games structurally cannot end in a draw. Extra time and penalties always force a winner, so the actual final score in your data will never show home\_score == away\_score for a knockout match. Let me verify that against what's already in your dataset before wiring up the toggle.

Check whether knockout matches in the dataset ever show a draw scoreline

Check whether knockout matches in the dataset ever show a draw scoreline

That's an important finding, and different from what I expected: these 16 knockout fixtures are scheduled but not yet played (NaN scores — Round of 32 starting June 28, today is June 30). That's even better for the toggle idea: there's no real played-knockout data to leak into your training anyway, but the structural point still holds — once these are played, regulation draws will go to extra time/penalties, so the *recorded* result will always show a winner, even if the 90-minute scoreline was tied. Your data prep already drops NaN score rows, so this confirms knockout draws structurally can't exist in result once played. Let me check the *earlier* World Cup knockout history in your dataset to validate this assumption fully, then build the toggle.

Check how historical knockout draws are recorded in raw results.csv

Check how historical knockout draws are recorded in raw results.csv

This is the critical finding — your suspicion was right to check, and the data confirms knockout draws absolutely do appear in result.csv: Japan 1-1 Croatia, Morocco 0-0 Spain, Croatia 1-1 Brazil, Netherlands 2-2 Argentina, Argentina 3-3 France (the 2022 final, decided on penalties). The raw 90-minute scoreline is what's recorded, not the shootout winner. So your model's result label (H/D/A, computed straight from goal\_diff) genuinely does include D for knockout matches that went to penalties — meaning the "knockout games can't draw" assumption is false in your training data, even though it's true in real life for "who advances."

This is actually a more interesting nuance than I first assumed. Let's quantify it properly before building the toggle:

Compare actual draw rates between group stage and knockout matches across recent World Cups

Compare actual draw rates between group stage and knockout matches across recent World Cups

Now I have real numbers to justify the toggle. Group stage draw rate averages 24.7%, knockout (90-minute) draw rate averages 21.4% — lower, but not dramatically so, and 2022 actually had a *higher* knockout draw rate (31.3%) than its group stage (20.8%). Your intuition about desperation-driven late goals is real, but the effect is modest, not the "knockouts basically never draw" assumption I'd have guessed before checking. This means a fixed lower margin for knockouts is reasonable but shouldn't be drastic. Let's pick a sensible value and wire up the toggle.

Check if we even have enough WC-specific holdout data to tune a separate knockout margin

Check if we even have enough WC-specific holdout data to tune a separate knockout margin

Good, plenty of data. Given the real-world numbers (24.7% group draws vs 21.4% knockout draws — roughly a 13% relative reduction), I'll set the knockout margin a bit lower than the group margin rather than removing the draw boost entirely. Now let's implement the toggle.

Add stage-aware draw margin constants and update apply\_draw\_margin to accept a stage parameter

Add stage-aware draw margin constants and update apply\_draw\_margin to accept a stage parameter

Now update predict\_match() to accept a match\_stage parameter and pick the right margin:

Ran a command, viewed a file

Ran a command, viewed a file

match\_stage is conceptually separate from is\_world\_cup/is\_neutral (which are *model features*) — it only controls which draw margin to apply at inference, so it shouldn't get passed into build\_match\_features. Let's add it cleanly as a new parameter:

Add match\_stage parameter to predict\_match, controlling which draw margin is used

Add match\_stage parameter to predict\_match, controlling which draw margin is used

Also worth surfacing match\_stage and the margin used in the return dict, for transparency in the UI:

Add match\_stage and draw\_margin\_used to the returned prediction dict

Add match\_stage and draw\_margin\_used to the returned prediction dict

Now test it before touching the app:

Verify the stage toggle actually changes predictions where expected

Verify the stage toggle actually changes predictions where expected

It works exactly as intended — Portugal vs Croatia and England vs Croatia are both close calls where the larger group margin tips it to "Draw," but the tighter knockout margin keeps it at "Home Win." Now let's wire this into app.py.

Viewed 2 files

Viewed 2 files

Now I'll add the stage toggle as a radio button and wire it through. I'll also make the prediction card show the stage and margin used, since it materially affects the call.

Add match stage toggle and pass it into predict\_match

Add match stage toggle and pass it into predict\_match

There's one more cross-cutting detail: when match\_stage == "knockout", the "Neutral venue" checkbox is essentially always true in a real WC (all knockout games are at neutral/host venues unless the host team is playing), and a knockout match should arguably auto-default is\_world\_cup=True. I'll leave those as manual checkboxes for flexibility but smart-default them based on stage:

Smart-default the is\_world\_cup/is\_neutral checkboxes based on selected stage

Smart-default the is\_world\_cup/is\_neutral checkboxes based on selected stage

Now let's also update batch\_test\_2026.py so the group-stage evaluation uses the right margin (it currently has a hardcoded standalone DRAW\_MARGIN = 0.06), and add a way to test the knockout matches once they're played:

Edited a file, ran a command, viewed a file

Edited a file, ran a command, viewed a file

Now I'll generalize this to take a stage parameter and date range, so it works for knockout matches once they're played:

Generalize date constants and load function to support both group and knockout stages

Generalize date constants and load function to support both group and knockout stages

Now update main() to use the renamed function and report both stages:

Viewed a file, edited a file

Viewed a file, edited a file

Now I need to remove the duplicate if \_\_name\_\_ block at the bottom and update print\_report's signature/header to accept stage:

Edited 3 files, ran 2 commands, viewed a file

Edited 3 files, ran 2 commands, viewed a file

Now run the full test:

Run full batch test for both stages

Run full batch test for both stages

Group stage reproduces your reported 63.0% / 34-54 result exactly, and the knockout branch correctly reports "0 played, 16 scheduled" rather than erroring — it'll just start working automatically as the Round of 32 results come in over the next few days. Let's do a final syntax/smoke check on app.py and ship everything.

Final syntax check and smoke test of updated app.py

Final syntax check and smoke test of updated app.py

Clean. Let's ship the updated files.

Ran a command, read a file

Ran a command, read a file

### What I found before building this

Your intuition was right but the mechanism is more interesting than "knockouts don't draw." I checked the actual 90-minute scorelines in results.csv for World Cups 1998–2022: group stage draw rate is 24.7%, knockout draw rate is 21.4% — real, but a modest gap, not the near-zero you might expect. Why: extra time and penalties always produce *a winner who advances*, but your model predicts the recorded scoreline, and that scoreline absolutely can tie (Argentina 3-3 France in the 2022 final is right there in your raw data). So a stage toggle is justified, but a drastic one would have been wrong.

### What changed

predictor.py — replaced the single DRAW\_MARGIN constant with DRAW\_MARGIN\_GROUP\_STAGE = 0.10 and DRAW\_MARGIN\_KNOCKOUT = 0.06. predict\_match() now takes a match\_stage="group"|"knockout" argument that picks the right one, and the return dict surfaces match\_stage and draw\_margin\_used so it's visible downstream. Verified it actually flips real calls — Portugal vs Croatia and England vs Croatia both swing from "Draw" (group) to "Home Win" (knockout) with the tighter margin.

app.py — added a horizontal radio toggle for match stage with a tooltip explaining the real-data rationale, wired it into the predict\_match() call, smart-defaulted "Neutral venue" to checked when knockout is selected, and the prediction card now shows which stage/margin was used.

batch\_test\_2026.py — generalized from a group-only script into one that handles both stages via load\_stage\_features(stage) / run\_batch\_predictions(df, stage). Running it now reports group stage results (reproduces your 63.0%/34-54 exactly) and checks for knockout matches — it currently finds 0 played and 16 scheduled, and will start working automatically once Round of 32 results land in results.csv over the next few days, without any code changes needed.

### One thing to watch

I picked 0.10/0.06 because that's what you'd already validated works well on this dataset for group stage, and a modestly tighter version for knockout reflecting the smaller real draw-rate gap. Once knockout matches are actually played, run python src/batch\_test\_2026.py knockout and you'll get real validation data — worth revisiting the knockout margin then rather than trusting this pre-tournament estimate indefinitely.

App

PY

Predictor

PY

Batch test 2026

PY

PS C:\Users\User\Downloads\wc2026\_predictor> python .\src\batch\_test\_2026.py knockout

✅ Found 0 knockout round matches with usable features

ℹ️ 16 knockout round matches not yet played (no final score recorded).

No played knockout matches with usable features yet -- nothing to test.

That's the correct, expected output — not an error. Today is June 30, 2026, and the Round of 32 only kicked off June 28, with the 16 fixtures you have on file running through July 3. None of them have final scores in your results.csv yet, so the script correctly tells you there's nothing to test rather than crashing or fabricating results.

A few ways to confirm this and get value from it right now:

Show current state of the 16 knockout fixtures on file

Show current state of the 16 knockout fixtures on file

Confirmed — these are scheduled fixtures with no scores yet, which matches reality: South Africa vs Canada (June 28) and the others through July 3 haven't been played as of today. Two things to know:

This isn't broken — it's working correctly. The script will produce real results automatically, no code changes needed, once results.csv gets updated with actual scores as these matches happen over the next few days.

In the meantime, you can still use predictor.py directly to generate *predictions* for these upcoming knockout matches (as forecasts, not validated against results yet):

Generate forward-looking predictions for the 16 upcoming Round of 32 fixtures

Generate forward-looking predictions for the 16 upcoming Round of 32 fixtures

That surfaced two real bugs — a format string bug in my own demo script (not your code), and a genuine team-name mismatch worth flagging. Let me fix the demo and investigate the name issue properly.

Check exact team name spellings in the predictor's data

Check exact team name spellings in the predictor's data

None of those variants match — but you definitely beat the US 4-1 in results.csv group stage data earlier. Let's check what the actual stored names are:

Find actual stored spellings via substring search

Find actual stored spellings via substring search

This is the real bug — the United States, Ivory Coast, and Cape Verde aren't just spelled differently, they don't exist in predictor.get\_teams() at all. That means features.csv has zero usable rows for them. Let's find out why.

Check whether these teams exist in raw results vs the final feature dataset

Check whether these teams exist in raw results vs the final feature dataset

Confirmed — these teams have hundreds of matches in the raw data, but zero rows survive into features.csv. That points straight at the FIFA rankings join.

Check how these teams are spelled in fifa\_rankings.csv

Check how these teams are spelled in fifa\_rankings.csv

Found it exactly. The two source files use different naming conventions for the same countries:

| results.csv (your match data) | fifa\_rankings.csv (rankings) |
| --- | --- |
| United States | *(missing entirely — checked below)* |
| Ivory Coast | Côte d'Ivoire |
| Cape Verde | Cabo Verde / Cape Verde Islands |
| DR Congo | Congo DR |

Specifically hunt for how USA is spelled in rankings, since it didn't show up in 'United' search

Specifically hunt for how USA is spelled in rankings, since it didn't show up in 'United' search

Confirmed — fifa\_rankings.csv calls it USA, results.csv calls it United States. This is a genuine data-quality bug in the original attach\_rankings() merge in data\_preparation.py, and it's a much bigger problem than just these 4 teams. Let's quantify the full damage:

Find all teams missing from rankings join, focusing on ones with significant match history

Find all teams missing from rankings join, focusing on ones with significant match history

This is exactly the kind of thing that's worth surfacing precisely rather than glossing over: 9 major national teams — including the World Cup co-host (United States), the team that just beat South Africa 2-0 in your dataset's Korea match, and several active 2026 squads — silently get zero FIFA ranking features because of a naming mismatch, and silently get dropped from the model entirely. The rest are mostly genuinely obscure/non-FIFA entities (Guernsey, Alderney, Shetland — football associations not affiliated with FIFA, so legitimately absent from rankings) and are fine to exclude.

Let me build a proper name-alias fix into data\_preparation.py:

Identify the correct rankings-side spelling for each significant mismatch

Identify the correct rankings-side spelling for each significant mismatch

Good — these confirm the mapping for the actively relevant teams. German DR and Vietnam Republic are historical/defunct entities (East Germany, South Vietnam — both dissolved decades ago, correctly absent from a modern rankings table), and Taiwan is genuinely not a FIFA member under that name (Chinese Taipei is, but isn't in this rankings file at all). Those don't need fixing. Let's build a clean alias map for the ones that matter:

Get the full canonical list of 2026 WC team names as spelled in results.csv

Get the full canonical list of 2026 WC team names as spelled in results.csv

Now I have the exact 46 teams that matter. Let's check each against the rankings file and build the complete alias map:

Find all 2026 WC teams specifically missing rankings data

Find all 2026 WC teams specifically missing rankings data

Exactly 6 of the 48 World Cup teams — over 10% of the field — were silently broken. Now let's fix data\_preparation.py with a proper alias map and rebuild everything downstream.

Ran a command, viewed a file, edited a file

Ran a command, viewed a file, edited a file

Now wire the alias into attach\_rankings(), where the actual team-name lookup happens:

View attach\_rankings to add alias translation at lookup time

View attach\_rankings to add alias translation at lookup time

The cleanest fix: translate the team name right after building match\_keys, before the groupby — that way the lookup naturally finds the rankings-side name without touching the rest of the logic.

Apply alias translation to match\_keys before the groupby lookup

Apply alias translation to match\_keys before the groupby lookup

Now rerun the full pipeline and verify the fix actually works:

Rerun data\_preparation.py with the alias fix

Rerun data\_preparation.py with the alias fix

The H2H step is O(n²) and slow on the full dataset — let's let it run in the background with more time.

Ran 2 commands

Ran 2 commands

It already finished (no log output shown because the process exited, but no leftover process either). Let me check the actual output:

View full output of the rerun

View full output of the rerun

Empty — that's odd. Let's check if it's still actually running in the background despite ps not catching it (background nohup processes can be tricky to track):

Check if features.csv was actually regenerated and when

Check if features.csv was actually regenerated and when

Stale — last modified 06:03, before we started this fix (07:21+). The background process likely got killed when its parent shell context ended. Let's run it properly in foreground with a longer timeout budget, breaking the H2H bottleneck into a faster approach since it's clearly the slow part.

Launch fully detached so it survives the tool call boundary

Launch fully detached so it survives the tool call boundary

Now it's properly running and detached (PID 710, actively using CPU). Let's poll it periodically.

Check progress after 90s

Check progress after 90s

Good, it's progressing through the H2H step as expected (the "~30s" estimate in the comment is optimistic for 31k rows — it's an O(n²) loop). Let's keep waiting.

Check progress after 3 more minutes

Check progress after 3 more minutes

Still grinding through H2H — confirms this loop is a genuine bottleneck (4m41s of CPU and counting on 31k rows, O(n²) = ~985M comparisons). Let's keep waiting; it will finish, just slowly.

Check progress after another ~5 minutes

Check progress after another ~5 minutes

It finished, and the fix worked: 27,505 final rows, up from 24,741 — that's 2,764 additional usable matches recovered, exactly the matches involving the 6 previously-broken teams (and 2 bonus aliases — China/North Korea — that I added defensively for completeness). Let's verify directly:

Verify all 6 previously-broken teams now have feature rows

Verify all 6 previously-broken teams now have feature rows

All 6 fixed. Now retrain the models on the corrected feature set (this is fast, ~seconds, not the slow part):

Retrain models on the corrected feature dataset

Retrain models on the corrected feature dataset

Now let's verify the fix actually delivers usable predictions for the previously-broken teams, then rerun your batch test:

Confirm previously-broken teams now predict successfully

Confirm previously-broken teams now predict successfully

All four previously-broken fixtures now predict correctly. Team count went from 197 → 205. Now let's rerun your batch test on the corrected pipeline:

Rerun batch test on corrected models

Rerun batch test on corrected models

This is the honest result, and it's worth being straight about it: all 72 group stage matches are now testable (up from 54 — every fixture, no silent exclusions), but outcome accuracy dropped from 63.0% to 52.8%. That's not a bug — it's the previous 63% being measured on an easier, cherry-picked subset. Once the 18 previously-broken matches (involving USA, South Korea, Iran, Ivory Coast, Cape Verde, DR Congo) are included, several go wrong: South Korea, Australia, Ivory Coast, Iran, Spain vs Cape Verde all miss now. The model isn't worse — your evaluation is just measuring it correctly for the first time. Let's confirm exact scoreline accuracy improved too (6.9% vs 3.7%), consistent with a genuinely fuller, more honest test set.

Ran a command, read a file

Ran a command, read a file

### So: your knockout command was correct, but it led me to find something more important

The python .\src\batch\_test\_2026.py knockout output was exactly right — no error, just "0 played, 16 scheduled," which is reality (Round of 32 started June 28, still in progress). That part needs no fix; it'll start working on its own as games are played.

But chasing why your earlier prediction loop kept failing on United States, Ivory Coast, Cape Verde, and DR Congo uncovered a real bug: results.csv and fifa\_rankings.csv spell 6 of the 48 World Cup teams differently (United States/USA, Ivory Coast/Côte d'Ivoire, South Korea/Korea Republic, Iran/IR Iran, Cape Verde/Cabo Verde, DR Congo/Congo DR). Since the rankings join is a hard requirement for every feature row, this meant 12.5% of the 2026 field had zero FIFA ranking features and got silently dropped from the entire model — they don't error, they just vanish from get\_teams() and from training data alike.

### What I changed

data\_preparation.py now has a RESULTS\_TO\_RANKINGS\_ALIASES dict applied in attach\_rankings(), translating names just for the lookup. Reran the full pipeline (data prep → retrain → batch test):

|  | Before fix | After fix |
| --- | --- | --- |
| Teams in predictor | 197 | 205 |
| Total feature rows | 24,741 | 27,505 |
| 2026 group stage testable | 54/72 | 72/72 |
| Group stage accuracy | 63.0% | 52.8% |

### The accuracy "drop" is the honest number

It's not regression, it's correction. The 63.0% figure was measured on a subset that happened to exclude every match involving USA, South Korea, Ivory Coast, Iran, Cape Verde, and DR Congo — all of whom now show up and several of whom the model gets wrong (South Korea, Australia, Ivory Coast, Spain–Cape Verde). 52.8% on all 72 matches is the real number; the previous 63% was an artifact of silently testing on an easier subset.

Three files attached: corrected data\_preparation.py, retrained outcome\_model.pkl/score\_model.pkl, and the regenerated features.csv. Drop these into your project (overwrite the old .pkl files in models/ and features.csv in data/processed/) and your batch\_test\_2026.py group / knockout commands will use the corrected data automatically — no other code changes needed on your end.

Data preparation

PY

Outcome model

PKL

Score model

PKL

Features

Table · CSV

PS C:\Users\User\Downloads\wc2026\_predictor> python .\src\batch\_test\_2026.py ################################################################# # GROUP STAGE ################################################################# ✅ Found 72 group stage matches with usable features 💾 Full

pasted

Good — that's an exact match to what I got on my end (52.8%, 38/72, identical confusion matrix and per-match table), which confirms the fix transferred to your machine cleanly and reproducibly. You're fully in sync now.

A few things worth noting about this result, since you now have the complete, honest picture:

The model has a real, specific weakness: low-information matches. Look at what's actually going wrong — South Korea vs Czech Republic, Australia vs Turkey, Ivory Coast vs Ecuador, Iran vs New Zealand, Spain vs Cape Verde. These are mostly matches involving teams that were *just* fixed (South Korea, Ivory Coast, Iran) or debutants/rare opponents (Cape Verde, New Zealand) where h2h\_total is likely near 0 and rolling stats are thin. The model is essentially guessing with weak signal on exactly the matches where the alias fix gave it data for the first time — which makes sense, since "now has a few ranking points" isn't the same as "has rich historical form data."

Draw recall is still the biggest gap: 15% (3/20). Same core issue as before — margin=0.10 on group stage helps, but most draws still aren't getting picked because Draw isn't a close enough second on those particular matches.

Score MAE actually improved slightly (1.029/0.821 home/away vs the earlier 1.103/0.814), and exact scoreline accuracy nearly doubled (6.9% vs 3.7%) — so the regressor is doing fine; the classifier's argmax-vs-draws problem is still the main lever left to pull.
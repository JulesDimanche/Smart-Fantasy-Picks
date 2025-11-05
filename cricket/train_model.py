import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# ==== Paths ====
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR,'data')
MODEL_DIR = os.path.join(BASE_DIR,'model')

os.makedirs(MODEL_DIR, exist_ok=True)

# ==== Load Datasets ====
data_merge = pd.read_csv(os.path.join(DATA_DIR, 'data_merge.csv'))
venue_agg = pd.read_csv(os.path.join(DATA_DIR, 'venue_aggregated.csv'))
recent_agg = pd.read_csv(os.path.join(DATA_DIR, 'recent_aggregated.csv'))
player_data = pd.read_csv(os.path.join(DATA_DIR, 'player_data_cleaned.csv'))

# ==== Helper Function ====
def merge_common(df):
    """Merge venue and recent averages to main data"""
    df = df.merge(venue_agg, on=['Player Name1', 'Venue'], how='left')
    df = df.merge(recent_agg, on='Player Name1', how='left')
    return df.fillna(0)

# ============================================================
# 1️⃣ Batsmen Model
# ============================================================
print("Training batsman model...")

batsmen = player_data[player_data["Role"].str.contains('Batter|Wicketkeeper', na=False)]
batsmen_data = merge_common(data_merge)
batsmen_data = batsmen_data[batsmen_data['Player Name1'].isin(batsmen['Player Name'])]

batsmen_features = [
    'Runs Scored', 'Fours', 'Sixes', 'avg_batting_points_x',
    'recent_avg_batting_x', 'Avg Runs per Over', 'Boundary %'
]

# Drop rows missing required features
batsmen_data = batsmen_data.dropna(subset=['Fantasy Points'])
X_batsmen = batsmen_data[batsmen_features].fillna(0)
y_batsmen = batsmen_data['Fantasy Points']

batsman_scaler = StandardScaler()
X_batsmen_scaled = batsman_scaler.fit_transform(X_batsmen)

batsman_model = xgb.XGBRegressor(
    n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42
)
batsman_model.fit(X_batsmen_scaled, y_batsmen)

joblib.dump(batsman_model, os.path.join(MODEL_DIR, 'batsman_model.joblib'))
joblib.dump(batsman_scaler, os.path.join(MODEL_DIR, 'batsman_scaler.joblib'))

print("✅ Batsman model and scaler saved.")

# ============================================================
# 2️⃣ Bowler Model
# ============================================================
print("Training bowler model...")

bowlers = player_data[player_data["Role"].str.contains('Bowler', na=False)]
bowlers_data = merge_common(data_merge)
bowlers_data = bowlers_data[bowlers_data['Player Name1'].isin(bowlers['Player Name'])]

bowler_features = [
    'Wickets', 'Economy', 'avg_bowling_points_x', 'recent_avg_bowling_x'
]

bowlers_data = bowlers_data.dropna(subset=['Fantasy Points'])
X_bowlers = bowlers_data[bowler_features].fillna(0)
y_bowlers = bowlers_data['Fantasy Points']

bowler_scaler = StandardScaler()
X_bowlers_scaled = bowler_scaler.fit_transform(X_bowlers)

bowler_model = xgb.XGBRegressor(
    n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42
)
bowler_model.fit(X_bowlers_scaled, y_bowlers)

joblib.dump(bowler_model, os.path.join(MODEL_DIR, 'bowler_model.joblib'))
joblib.dump(bowler_scaler, os.path.join(MODEL_DIR, 'bowler_scaler.joblib'))

print("✅ Bowler model and scaler saved.")

# ============================================================
# 3️⃣ Allrounder Model
# ============================================================
print("Training allrounder model...")

allrounders = player_data[player_data["Role"].str.contains('Allrounder', na=False)]
allrounders_data = merge_common(data_merge)
allrounders_data = allrounders_data[allrounders_data['Player Name1'].isin(allrounders['Player Name'])]

allrounder_features = [
    'avg_batting_points_x', 'recent_avg_batting_x',
    'avg_bowling_points_x', 'recent_avg_bowling_x'
]

allrounders_data = allrounders_data.dropna(subset=['Fantasy Points'])
X_allrounders = allrounders_data[allrounder_features].fillna(0)
y_allrounders = allrounders_data['Fantasy Points']

allrounder_scaler = StandardScaler()
X_allrounders_scaled = allrounder_scaler.fit_transform(X_allrounders)

allrounder_model = xgb.XGBRegressor(
    n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42
)
allrounder_model.fit(X_allrounders_scaled, y_allrounders)

joblib.dump(allrounder_model, os.path.join(MODEL_DIR, 'allrounder_model.joblib'))
joblib.dump(allrounder_scaler, os.path.join(MODEL_DIR, 'allrounder_scaler.joblib'))

print("✅ Allrounder model and scaler saved.")
print("\n🎯 Training completed successfully!")

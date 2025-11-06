from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from typing import List, Tuple

# Flask Blueprint
cricket_bp = Blueprint('cricket', __name__, 
                      template_folder='templates',
                      static_folder='static',
                      static_url_path='/cricket/static')

class FantasyTeamSelector:
    def __init__(self, data_merge_path: str, venue_agg_path: str, recent_agg_path: str, 
                 player_data_path: str, matchup_data_path: str, model_dir: str):
        """Initialize with pre-trained models and data paths."""
        # Load datasets
        self.player_data = pd.read_csv(player_data_path)
        self.matchup_data = pd.read_csv(matchup_data_path)
        self.venue_agg = pd.read_csv(venue_agg_path)
        self.recent_agg = pd.read_csv(recent_agg_path)
        self.data_merge = pd.read_csv(data_merge_path)

        # Load saved models and scalers
        self.batsman_model = joblib.load(os.path.join(model_dir, 'batsman_model.joblib'))
        self.batsman_scaler = joblib.load(os.path.join(model_dir, 'batsman_scaler.joblib'))

        self.bowler_model = joblib.load(os.path.join(model_dir, 'bowler_model.joblib'))
        self.bowler_scaler = joblib.load(os.path.join(model_dir, 'bowler_scaler.joblib'))

        self.allrounder_model = joblib.load(os.path.join(model_dir, 'allrounder_model.joblib'))
        self.allrounder_scaler = joblib.load(os.path.join(model_dir, 'allrounder_scaler.joblib'))

        # Preprocess data (but no training)
        self._preprocess_data()

    def _preprocess_data(self):
        """Prepare merged dataframes for each role."""
        def merge_common(df):
            return df.merge(self.venue_agg, on=['Player Name1', 'Venue'], how='left') \
                     .merge(self.recent_agg, on='Player Name1', how='left') \
                     .fillna(0)

        # All datasets
        self.batsmen_data = merge_common(self.data_merge)
        self.bowlers_data = merge_common(self.data_merge)
        self.allrounders_data = merge_common(self.data_merge)

        # Role lists
        self.batsmen_list = self.player_data[self.player_data["Role"].str.contains('Batter|Wicketkeeper', na=False)]['Player Name'].tolist()
        self.bowlers_list = self.player_data[self.player_data["Role"].str.contains('Bowler', na=False)]['Player Name'].tolist()
        self.allrounders_list = self.player_data[self.player_data["Role"].str.contains('Allrounder', na=False)]['Player Name'].tolist()

        # Feature sets
        self.batsmen_features = ['Runs Scored', 'Fours', 'Sixes', 'avg_batting_points_x',
                                 'recent_avg_batting_x', 'Avg Runs per Over', 'Boundary %']
        self.bowlers_features = ['Wickets', 'Economy', 'avg_bowling_points_x', 'recent_avg_bowling_x']
        self.allrounders_features = ['avg_batting_points_x', 'recent_avg_batting_x',
                                     'avg_bowling_points_x', 'recent_avg_bowling_x']

    # ============================================================
    # Team Selection
    # ============================================================
    def select_best_team(self, team1: List[str], team2: List[str], venue: str,
                         batsmen_count: int = 5, bowlers_count: int = 5, allrounders_count: int = 1):
        best_batsmen = self._select_best_batsmen(team1, team2, venue)[:batsmen_count]
        best_bowlers = self._select_best_bowlers(team1, team2, venue)[:bowlers_count]
        best_allrounders = self._select_best_allrounders(team1, team2, venue)[:allrounders_count]

        combined = best_batsmen + best_bowlers + best_allrounders
        combined.sort(key=lambda x: x[1], reverse=True)

        return {
            'batsmen': [x[0] for x in best_batsmen],
            'bowlers': [x[0] for x in best_bowlers],
            'allrounders': [x[0] for x in best_allrounders],
            'combined_team': [x[0] for x in combined]
        }

    # ============================================================
    # Role-based Selection
    # ============================================================
    def _select_best_batsmen(self, team1, team2, venue):
        batsmen = [p for p in team1 + team2 if p in self.batsmen_list]
        scores = {}
        for player in batsmen:
            df = self.batsmen_data[self.batsmen_data['Player Name1'] == player]
            if df.empty:
                continue
            X = df[self.batsmen_features].fillna(0)
            X_scaled = self.batsman_scaler.transform(X)
            perf = self.batsman_model.predict(X_scaled)[0]

            matchup = self._get_batsman_matchup_score(player, team2)
            venue_score = df[df['Venue'] == venue]['avg_batting_points_x'].mean()
            recent_score = df['recent_avg_batting_x'].mean()

            total = perf + 0.4 * venue_score + 0.3 * recent_score + 0.3 * matchup
            scores[player] = total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _select_best_bowlers(self, team1, team2, venue):
        bowlers = [p for p in team1 + team2 if p in self.bowlers_list]
        scores = {}
        for player in bowlers:
            df = self.bowlers_data[self.bowlers_data['Player Name1'] == player]
            if df.empty:
                continue
            X = df[self.bowlers_features].fillna(0)
            X_scaled = self.bowler_scaler.transform(X)
            perf = self.bowler_model.predict(X_scaled)[0]

            matchup = self._get_bowler_matchup_score(player, team2)
            venue_score = df[df['Venue'] == venue]['avg_bowling_points_x'].mean()
            recent_score = df['recent_avg_bowling_x'].mean()

            total = perf + 0.4 * venue_score + 0.3 * recent_score + 0.3 * matchup
            scores[player] = total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _select_best_allrounders(self, team1, team2, venue):
        allrounders = [p for p in team1 + team2 if p in self.allrounders_list]
        scores = {}
        for player in allrounders:
            df = self.allrounders_data[self.allrounders_data['Player Name1'] == player]
            if df.empty:
                continue
            X = df[self.allrounders_features].fillna(0)
            X_scaled = self.allrounder_scaler.transform(X)
            perf = self.allrounder_model.predict(X_scaled)[0]

            matchup = self._get_allrounder_matchup_score(player, team2)
            venue_score = df[df['Venue'] == venue]['avg_batting_points_x'].mean()
            recent_score = df['recent_avg_batting_x'].mean()

            total = perf + 0.2 * venue_score + 0.2 * recent_score + 0.3 * matchup
            scores[player] = total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ============================================================
    # Matchup Calculations
    # ============================================================
    def _get_batsman_matchup_score(self, batsman, opposition):
        scores = self.matchup_data[
            (self.matchup_data['Batsman'] == batsman) &
            (self.matchup_data['Bowler'].isin(opposition))
        ]['Batsman Points']
        return scores.mean() if not scores.empty else 0

    def _get_bowler_matchup_score(self, bowler, opposition):
        scores = self.matchup_data[
            (self.matchup_data['Bowler'] == bowler) &
            (self.matchup_data['Batsman'].isin(opposition))
        ]['Bowler Points']
        return scores.mean() if not scores.empty else 0

    def _get_allrounder_matchup_score(self, player, opposition):
        scores = self.matchup_data[
            (self.matchup_data['Batsman'] == player) &
            (self.matchup_data['Bowler'].isin(opposition))
        ]['Batsman Points']
        return scores.mean() if not scores.empty else 0


# ============================================================
# Flask Blueprint Integration
# ============================================================
selector = None

@cricket_bp.record_once
def on_load(state):
    global selector
    # Resolve relative to the app root so it works on Vercel and locally
    base_dir = state.app.root_path
    cricket_dir = os.path.join(base_dir, 'cricket')
    cricket_data_dir = os.path.join(cricket_dir, 'data')
    model_dir = os.path.join(cricket_dir, 'model')

    data_paths = {
        'data_merge_path': os.path.join(cricket_data_dir, 'data_merge.csv'),
        'venue_agg_path': os.path.join(cricket_data_dir, 'venue_aggregated.csv'),
        'recent_agg_path': os.path.join(cricket_data_dir, 'recent_aggregated.csv'),
        'player_data_path': os.path.join(cricket_data_dir, 'player_data_cleaned.csv'),
        'matchup_data_path': os.path.join(cricket_data_dir, 'cleaned_matchup.csv'),
        'model_dir': model_dir
    }

    selector = FantasyTeamSelector(**data_paths)

@cricket_bp.route('/', methods=['GET', 'POST'])
def cricket_home():
    if request.method == 'POST':
        # ✅ Get the players correctly
        team1 = [t.strip() for t in request.form.get('team1_players', '').split(',') if t.strip()]
        team2 = [t.strip() for t in request.form.get('team2_players', '').split(',') if t.strip()]
        venue = request.form.get('venue', '')
        batsmen_count = int(request.form.get('batsmen_count', 5))
        bowlers_count = int(request.form.get('bowlers_count', 5))
        allrounders_count = int(request.form.get('allrounders_count', 1))

        results = selector.select_best_team(
            team1, team2, venue,
            batsmen_count=batsmen_count,
            bowlers_count=bowlers_count,
            allrounders_count=allrounders_count
        )

        # ✅ Pass input players to the template
        input_players = team1 + team2

        return render_template(
            'cricket.html',
            results=results,
            input_players=input_players,
            batsmen_count=batsmen_count,
            bowlers_count=bowlers_count,
            allrounders_count=allrounders_count,
            team1=', '.join(team1),
            team2=', '.join(team2),
            venue=venue
        )

    return render_template('cricket.html')

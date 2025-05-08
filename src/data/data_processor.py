import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def process_match_data(self, home_team_data: Dict[str, Any], away_team_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maç verilerini işler ve tahmin modeli için hazırlar."""
        try:
            # Son form durumu
            home_form = self._calculate_form(home_team_data['recent_matches'])
            away_form = self._calculate_form(away_team_data['recent_matches'])
            
            # Gol istatistikleri
            home_goals_stats = self._calculate_goal_stats(home_team_data['recent_matches'])
            away_goals_stats = self._calculate_goal_stats(away_team_data['recent_matches'])
            
            # Sezon performansı
            home_season = self._process_season_stats(home_team_data['season_stats'])
            away_season = self._process_season_stats(away_team_data['season_stats'])
            
            # Oyuncu performansları
            home_player_impact = self._calculate_player_impact(home_team_data['player_stats'])
            away_player_impact = self._calculate_player_impact(away_team_data['player_stats'])
            
            processed_data = {
                'home_form': home_form,
                'away_form': away_form,
                'home_goals_scored_avg': home_goals_stats['scored_avg'],
                'home_goals_conceded_avg': home_goals_stats['conceded_avg'],
                'away_goals_scored_avg': away_goals_stats['scored_avg'],
                'away_goals_conceded_avg': away_goals_stats['conceded_avg'],
                'home_season_points': home_season['points'],
                'away_season_points': away_season['points'],
                'home_player_impact': home_player_impact,
                'away_player_impact': away_player_impact
            }
            
            # Verileri normalize et
            normalized_data = self._normalize_data(processed_data)
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error processing match data: {str(e)}")
            raise
            
    def _calculate_form(self, recent_matches: List[Dict]) -> float:
        """Son maçlardaki form durumunu hesaplar."""
        form_points = []
        for match in recent_matches:
            if match['winner'] == 'HOME_TEAM':
                points = 3 if match['home_team'] else 0
            elif match['winner'] == 'AWAY_TEAM':
                points = 3 if not match['home_team'] else 0
            else:
                points = 1
            form_points.append(points)
        
        # Son maçlara daha fazla ağırlık ver
        weights = np.array([1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        weighted_form = np.average(form_points, weights=weights[:len(form_points)])
        return weighted_form
        
    def _calculate_goal_stats(self, matches: List[Dict]) -> Dict[str, float]:
        """Gol istatistiklerini hesaplar."""
        scored = []
        conceded = []
        
        for match in matches:
            if match['home_team']:
                scored.append(match['home_score'])
                conceded.append(match['away_score'])
            else:
                scored.append(match['away_score'])
                conceded.append(match['home_score'])
                
        return {
            'scored_avg': np.mean(scored),
            'conceded_avg': np.mean(conceded)
        }
        
    def _process_season_stats(self, season_stats: Dict) -> Dict:
        """Sezon istatistiklerini işler."""
        return {
            'points': season_stats.get('points', 0),
            'goals_scored': season_stats.get('goals_scored', 0),
            'goals_conceded': season_stats.get('goals_conceded', 0),
            'clean_sheets': season_stats.get('clean_sheets', 0)
        }
        
    def _calculate_player_impact(self, player_stats: List[Dict]) -> float:
        """Oyuncu performanslarının takım üzerindeki etkisini hesaplar."""
        # Oyuncu istatistiklerini değerlendir ve bir etki skoru hesapla
        impact_score = 0
        if player_stats:
            # Oyuncu performanslarını değerlendir
            pass
        return impact_score
        
    def _normalize_data(self, data: Dict) -> Dict:
        """Verileri normalize eder."""
        features = list(data.keys())
        values = np.array([data[f] for f in features]).reshape(1, -1)
        normalized_values = self.scaler.fit_transform(values)
        
        return dict(zip(features, normalized_values[0])) 
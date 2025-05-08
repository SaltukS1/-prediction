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
            home_form = self._calculate_form(home_team_data['recent_matches'], True)
            away_form = self._calculate_form(away_team_data['recent_matches'], False)
            
            # Gol istatistikleri
            home_goals_stats = self._calculate_goal_stats(home_team_data['recent_matches'], True)
            away_goals_stats = self._calculate_goal_stats(away_team_data['recent_matches'], False)
            
            # Sezon performansı
            home_season = self._process_season_stats(home_team_data['season_stats'])
            away_season = self._process_season_stats(away_team_data['season_stats'])
            
            # Oyuncu performansları (örnek değerler)
            home_player_impact = 0.8
            away_player_impact = 0.7
            
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
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing match data: {str(e)}")
            raise
            
    def _calculate_form(self, recent_matches: List[Dict], is_home: bool) -> float:
        """Son maçlardaki form durumunu hesaplar."""
        if not recent_matches:
            return 0.5  # Veri yoksa ortalama değer
            
        form_points = []
        for match in recent_matches:
            is_target_home = match.get('home_team_is_target', match['home_team'] == match['away_team'])
            
            if match['winner'] == 'HOME_TEAM':
                points = 3 if is_target_home else 0
            elif match['winner'] == 'AWAY_TEAM':
                points = 3 if not is_target_home else 0
            else:
                points = 1
            form_points.append(points)
        
        # Son maçlara daha fazla ağırlık ver
        weights = np.array([1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        if len(form_points) < len(weights):
            weights = weights[:len(form_points)]
            
        weighted_form = np.average(form_points, weights=weights[:len(form_points)])
        
        # Ev sahibi avantajı ekle
        if is_home:
            weighted_form *= 1.1
            
        return min(weighted_form, 5.0)  # Maksimum değer 5.0
        
    def _calculate_goal_stats(self, matches: List[Dict], is_home: bool) -> Dict[str, float]:
        """Gol istatistiklerini hesaplar."""
        if not matches:
            return {'scored_avg': 1.0, 'conceded_avg': 1.0}  # Veri yoksa ortalama değerler
            
        scored = []
        conceded = []
        
        for match in matches:
            is_target_home = match.get('home_team_is_target', match['home_team'] == match['away_team'])
            
            if is_target_home:
                scored.append(match['home_score'])
                conceded.append(match['away_score'])
            else:
                scored.append(match['away_score'])
                conceded.append(match['home_score'])
                
        # Ev sahibi avantajı/dezavantajı
        factor = 1.1 if is_home else 0.9
                
        return {
            'scored_avg': np.mean(scored) * factor if scored else 1.0,
            'conceded_avg': np.mean(conceded) / factor if conceded else 1.0
        }
        
    def _process_season_stats(self, season_stats: Dict) -> Dict:
        """Sezon istatistiklerini işler."""
        if not season_stats:
            return {
                'points': 50,
                'goals_scored': 35,
                'goals_conceded': 35,
                'clean_sheets': 5
            }
            
        return {
            'points': season_stats.get('points', 50),
            'goals_scored': season_stats.get('goals_scored', 35),
            'goals_conceded': season_stats.get('goals_conceded', 35),
            'clean_sheets': season_stats.get('clean_sheets', 5)
        }
        
    def _normalize_data(self, data: Dict) -> Dict:
        """Verileri normalize eder."""
        features = list(data.keys())
        values = np.array([data[f] for f in features]).reshape(1, -1)
        
        # 0-1 aralığına normalize et (basitlik için StandardScaler yerine manuel)
        normalized = {}
        for i, feature in enumerate(features):
            val = values[0, i]
            if 'goals' in feature:
                normalized[feature] = val / 5.0  # Gol ortalamaları 0-5 arasında normalleştir
            elif 'form' in feature:
                normalized[feature] = val / 3.0  # Form 0-3 arasında normalleştir
            elif 'points' in feature:
                normalized[feature] = val / 100.0  # Puanlar 0-100 arasında normalleştir
            elif 'impact' in feature:
                normalized[feature] = val  # Impact zaten 0-1 arasında
            else:
                normalized[feature] = val / 10.0  # Diğerleri için basit bölme
                
        return normalized 
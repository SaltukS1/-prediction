import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
from typing import Dict, List, Any
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.base_url = "https://api.football-data.org/v2"
        load_dotenv()
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY", "3349986edb544bf3abe89d524466affb")
        self.headers = {"X-Auth-Token": self.api_key}
        
    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Bir takımın son maçlarındaki istatistiklerini toplar."""
        try:
            # Gerçek API yerine örnek veri döndür
            return self._get_mock_team_stats(team_name)
        except Exception as e:
            logger.error(f"Error collecting stats for {team_name}: {str(e)}")
            raise
    
    def _get_mock_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Test için takım istatistikleri döndürür."""
        # Rastgele örnek veriler oluştur
        recent_matches = self._generate_mock_matches(team_name, 10)
        
        # Eğer çok güçlü bir takımsa daha yüksek değerler ver
        strong_teams = ["Barcelona", "Real Madrid", "Bayern Munich", "Manchester City", 
                       "Liverpool", "PSG", "Galatasaray", "Fenerbahçe", "Beşiktaş"]
        
        strength_factor = 1.5 if team_name in strong_teams else 1.0
        
        return {
            "recent_matches": recent_matches,
            "season_stats": {
                "points": random.randint(35, 85) * strength_factor,
                "goals_scored": random.randint(30, 90) * strength_factor,
                "goals_conceded": random.randint(15, 50) / strength_factor,
                "clean_sheets": random.randint(5, 20) * strength_factor
            },
            "player_stats": [],  # Basitleştirme için boş bırakıyoruz
            "head_to_head": []   # Basitleştirme için boş bırakıyoruz
        }
    
    def _generate_mock_matches(self, team_name: str, count: int = 10) -> List[Dict]:
        """Test için rastgele maç verileri oluşturur."""
        teams = ["Barcelona", "Real Madrid", "Bayern Munich", "Manchester City", 
                "Liverpool", "PSG", "Inter Milan", "Juventus", "Arsenal", 
                "Chelsea", "Atletico Madrid", "Borussia Dortmund", "AC Milan",
                "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"]
        
        matches = []
        for i in range(count):
            # Bizim takım ev sahibi mi yoksa deplasman mı?
            is_home = random.choice([True, False])
            
            # Rakip takım
            opponent = random.choice([t for t in teams if t != team_name])
            
            home_team = team_name if is_home else opponent
            away_team = opponent if is_home else team_name
            
            # Skor
            home_score = random.randint(0, 4)
            away_score = random.randint(0, 3)
            
            # Kazanan
            if home_score > away_score:
                winner = "HOME_TEAM"
            elif away_score > home_score:
                winner = "AWAY_TEAM"
            else:
                winner = "DRAW"
            
            match = {
                'date': f"2022-{random.randint(1, 12)}-{random.randint(1, 28)}",
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'winner': winner,
                'status': "FINISHED",
                'home_team_is_target': is_home
            }
            matches.append(match)
        
        return matches
    
    def _process_matches(self, matches: List[Dict]) -> List[Dict]:
        """Ham maç verilerini işler ve analiz için hazırlar."""
        processed_matches = []
        for match in matches:
            processed_match = {
                'date': match['utcDate'],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_score': match['score']['fullTime']['homeTeam'],
                'away_score': match['score']['fullTime']['awayTeam'],
                'winner': match['score']['winner'],
                'status': match['status']
            }
            processed_matches.append(processed_match)
        return processed_matches
    
    def get_odds_data(self, match_id: str) -> Dict:
        """Maç için bahis oranlarını getirir."""
        # Rastgele bahis oranları döndür
        return {
            "home_win": round(random.uniform(1.2, 4.5), 2),
            "draw": round(random.uniform(2.0, 5.0), 2),
            "away_win": round(random.uniform(1.5, 6.0), 2)
        } 
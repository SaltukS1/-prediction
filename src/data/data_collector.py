import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
from typing import Dict, List, Any, Tuple
import logging
import os
from dotenv import load_dotenv
import json
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.base_url = "https://api.football-data.org/v2"
        load_dotenv()
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY", "3349986edb544bf3abe89d524466affb")
        self.headers = {"X-Auth-Token": self.api_key}
        
        # Takım performans veritabanı - gerçek API'ye bağlanıldığında güncel veriler kullanılır
        self.team_database = TeamDatabase()
        
    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Bir takımın son maçlarındaki istatistiklerini toplar."""
        try:
            # Gerçek API yerine takımın sezon performansını içeren gelişmiş simülasyon verisi döndür
            team_data = self.team_database.get_team_data(team_name)
            recent_matches = self.team_database.get_recent_matches(team_name)
            
            return {
                "recent_matches": recent_matches,
                "season_stats": team_data["season_stats"],
                "team_stats": team_data["team_stats"],
                "player_stats": team_data["player_stats"],
                "head_to_head": []  # İleride geliştirilebilir
            }
        except Exception as e:
            logger.error(f"Error collecting stats for {team_name}: {str(e)}")
            # Hata durumunda yedek veri oluştur
            return self._get_mock_team_stats(team_name)
    
    def _get_mock_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Test için takım istatistikleri döndürür."""
        # Takım adına göre kaba bir güç faktörü oluştur
        team_tiers = {
            # Süper Lig
            "Galatasaray": 0.85, "Fenerbahçe": 0.85, "Beşiktaş": 0.80, "Trabzonspor": 0.78,
            # Premier Lig
            "Manchester City": 0.95, "Liverpool": 0.90, "Arsenal": 0.88, "Manchester United": 0.85,
            # La Liga
            "Real Madrid": 0.95, "Barcelona": 0.90, "Atletico Madrid": 0.85,
            # Bundesliga
            "Bayern Munich": 0.95, "Borussia Dortmund": 0.88,
            # Serie A
            "Inter Milan": 0.88, "Juventus": 0.85, "AC Milan": 0.85,
            # Ligue 1
            "PSG": 0.90
        }
        
        # Takım değerini belirle (varsayılan 0.75)
        team_strength = team_tiers.get(team_name, 0.75)
        
        # Takımın son 10 maç performansını simüle et
        recent_matches = self._generate_mock_matches(team_name, 10, team_strength)
        
        # Ligdeki sıra ve puan durumunu simüle et
        if team_strength > 0.85:
            league_position = random.randint(1, 4)
            points = random.randint(75, 95)
        elif team_strength > 0.8:
            league_position = random.randint(3, 6)
            points = random.randint(65, 80)
        else:
            league_position = random.randint(5, 18)
            points = random.randint(35, 65)
        
        # Sezon istatistiklerini oluştur
        matches_played = random.randint(30, 38)
        wins = random.randint(int(matches_played * team_strength * 0.7), int(matches_played * team_strength * 0.9))
        draws = random.randint(3, 10)
        losses = matches_played - wins - draws
        
        goals_scored = wins * random.uniform(1.5, 2.5) + draws * random.uniform(0.7, 1.3)
        goals_conceded = losses * random.uniform(1.3, 2.0) + draws * random.uniform(0.7, 1.3)
        
        return {
            "recent_matches": recent_matches,
            "season_stats": {
                "league_position": league_position,
                "points": points,
                "matches_played": matches_played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_scored": round(goals_scored),
                "goals_conceded": round(goals_conceded),
                "goals_scored_avg": round(goals_scored / matches_played, 2),
                "goals_conceded_avg": round(goals_conceded / matches_played, 2),
                "clean_sheets": random.randint(3, wins),
                "form": round(self._calculate_form(recent_matches) * 100)
            },
            "team_stats": {
                "attack_strength": round(team_strength * 100),
                "defense_strength": round(team_strength * 95),
                "home_advantage": random.randint(5, 15),
                "consistency": random.randint(70, 90)
            },
            "player_stats": self._generate_mock_player_stats(team_name, team_strength)
        }
    
    def _generate_mock_matches(self, team_name: str, count: int = 10, team_strength: float = 0.75) -> List[Dict]:
        """Test için rastgele maç verileri oluşturur."""
        teams = [
            "Barcelona", "Real Madrid", "Bayern Munich", "Manchester City", 
            "Liverpool", "PSG", "Inter Milan", "Juventus", "Arsenal", 
            "Chelsea", "Atletico Madrid", "Borussia Dortmund", "AC Milan",
            "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"
        ]
        
        matches = []
        for i in range(count):
            # Bizim takım ev sahibi mi yoksa deplasman mı?
            is_home = random.choice([True, False])
            
            # Rakip takım
            opponent = random.choice([t for t in teams if t != team_name])
            
            home_team = team_name if is_home else opponent
            away_team = opponent if is_home else team_name
            
            # Maç sonucunu takım gücüne göre ayarla
            if is_home:
                win_prob = team_strength + 0.1  # Ev avantajı
                lose_prob = 1 - win_prob - 0.2  # Beraberlik şansı 0.2
            else:
                win_prob = team_strength - 0.1  # Deplasman dezavantajı
                lose_prob = 1 - win_prob - 0.2  # Beraberlik şansı 0.2
            
            # Sonuç
            result = random.choices(
                ["HOME_TEAM", "DRAW", "AWAY_TEAM"],
                weights=[win_prob if is_home else lose_prob, 0.2, lose_prob if is_home else win_prob]
            )[0]
            
            # Skor
            if result == "HOME_TEAM":
                home_score = random.randint(1, 4)
                away_score = random.randint(0, home_score - 1)
            elif result == "AWAY_TEAM":
                away_score = random.randint(1, 4)
                home_score = random.randint(0, away_score - 1)
            else:  # Beraberlik
                home_score = random.randint(0, 2)
                away_score = home_score
            
            # Rastgele bir tarih (son 60 gün içinde)
            days_ago = i * 7  # Her maç bir hafta önce
            match_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            match = {
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'winner': result,
                'status': "FINISHED",
                'home_team_is_target': is_home
            }
            matches.append(match)
        
        return matches
    
    def _generate_mock_player_stats(self, team_name: str, team_strength: float) -> List[Dict]:
        """Takım için rastgele oyuncu istatistikleri oluşturur."""
        # Bu metot, gerçek bir API'den oyuncu istatistikleri alınamadığında kullanılır
        team_players = []
        return team_players
    
    def _calculate_form(self, matches: List[Dict]) -> float:
        """Son maçlara göre takımın form puanını hesaplar (0-1 arası)."""
        if not matches:
            return 0.5
        
        form_points = 0
        total_possible = 0
        
        # Her maç için puan hesapla (daha yakın tarihli maçlar daha önemli)
        for i, match in enumerate(matches):
            match_weight = 1.0 - (i / len(matches) * 0.5)  # 1 ile 0.5 arasında bir ağırlık
            
            is_target_home = match.get('home_team_is_target', False)
            
            if (match['winner'] == 'HOME_TEAM' and is_target_home) or (match['winner'] == 'AWAY_TEAM' and not is_target_home):
                form_points += 3 * match_weight  # Galibiyet
            elif match['winner'] == 'DRAW':
                form_points += 1 * match_weight  # Beraberlik
            
            total_possible += 3 * match_weight
        
        # Formu 0-1 arasında normalize et
        return form_points / total_possible if total_possible > 0 else 0.5

    def get_odds_data(self, match_id: str) -> Dict:
        """Maç için bahis oranlarını getirir."""
        # Rastgele bahis oranları döndür
        return {
            "home_win": round(random.uniform(1.2, 4.5), 2),
            "draw": round(random.uniform(2.0, 5.0), 2),
            "away_win": round(random.uniform(1.5, 6.0), 2)
        }

class TeamDatabase:
    """Takımların tarihsel performans verilerini sağlayan veritabanı sınıfı."""
    
    def __init__(self):
        # Takımların performans verileri
        self.teams_data = self._initialize_teams_data()
        
    def _initialize_teams_data(self) -> Dict[str, Dict]:
        """Takımların performans verilerini yükler veya oluşturur."""
        # Gerçek veritabanından veya API'den çekilebilir
        # Şu an için statik veriler kullanılıyor
        
        teams_data = {}
        
        # Süper Lig takımları
        teams_data["Galatasaray"] = self._create_team_data(
            league="Süper Lig",
            position=1,
            form=0.92,
            goals_scored_avg=2.4,
            goals_conceded_avg=0.8,
            home_form=0.95,
            away_form=0.85,
            home_goals_avg=2.7,
            away_goals_avg=2.1,
            strength=0.88
        )
        
        teams_data["Fenerbahçe"] = self._create_team_data(
            league="Süper Lig",
            position=2,
            form=0.9,
            goals_scored_avg=2.6,
            goals_conceded_avg=1.0,
            home_form=0.92,
            away_form=0.85,
            home_goals_avg=2.9,
            away_goals_avg=2.2,
            strength=0.87
        )
        
        teams_data["Beşiktaş"] = self._create_team_data(
            league="Süper Lig",
            position=3,
            form=0.75,
            goals_scored_avg=1.9,
            goals_conceded_avg=1.3,
            home_form=0.8,
            away_form=0.7,
            home_goals_avg=2.2,
            away_goals_avg=1.6,
            strength=0.78
        )
        
        teams_data["Trabzonspor"] = self._create_team_data(
            league="Süper Lig",
            position=5,
            form=0.7,
            goals_scored_avg=1.7,
            goals_conceded_avg=1.4,
            home_form=0.82,
            away_form=0.65,
            home_goals_avg=2.0,
            away_goals_avg=1.3,
            strength=0.75
        )
        
        # Premier Lig takımları
        teams_data["Manchester City"] = self._create_team_data(
            league="Premier League",
            position=1,
            form=0.9,
            goals_scored_avg=2.7,
            goals_conceded_avg=0.9,
            home_form=0.92,
            away_form=0.88,
            home_goals_avg=3.0,
            away_goals_avg=2.3,
            strength=0.92
        )
        
        teams_data["Liverpool"] = self._create_team_data(
            league="Premier League",
            position=2,
            form=0.85,
            goals_scored_avg=2.4,
            goals_conceded_avg=1.0,
            home_form=0.9,
            away_form=0.82,
            home_goals_avg=2.7,
            away_goals_avg=2.1,
            strength=0.88
        )
        
        teams_data["Arsenal"] = self._create_team_data(
            league="Premier League",
            position=3,
            form=0.83,
            goals_scored_avg=2.2,
            goals_conceded_avg=0.9,
            home_form=0.87,
            away_form=0.8,
            home_goals_avg=2.5,
            away_goals_avg=1.9,
            strength=0.86
        )
        
        # La Liga takımları
        teams_data["Real Madrid"] = self._create_team_data(
            league="La Liga",
            position=1,
            form=0.9,
            goals_scored_avg=2.5,
            goals_conceded_avg=0.8,
            home_form=0.92,
            away_form=0.88,
            home_goals_avg=2.8,
            away_goals_avg=2.2,
            strength=0.92
        )
        
        teams_data["Barcelona"] = self._create_team_data(
            league="La Liga",
            position=2,
            form=0.82,
            goals_scored_avg=2.3,
            goals_conceded_avg=1.0,
            home_form=0.88,
            away_form=0.78,
            home_goals_avg=2.6,
            away_goals_avg=2.0,
            strength=0.86
        )
        
        # Bundesliga takımları
        teams_data["Bayern Munich"] = self._create_team_data(
            league="Bundesliga",
            position=1,
            form=0.87,
            goals_scored_avg=3.0,
            goals_conceded_avg=1.2,
            home_form=0.9,
            away_form=0.85,
            home_goals_avg=3.3,
            away_goals_avg=2.7,
            strength=0.9
        )
        
        # Serie A takımları
        teams_data["Inter Milan"] = self._create_team_data(
            league="Serie A",
            position=1,
            form=0.85,
            goals_scored_avg=2.2,
            goals_conceded_avg=0.9,
            home_form=0.88,
            away_form=0.82,
            home_goals_avg=2.5,
            away_goals_avg=1.9,
            strength=0.85
        )
        
        return teams_data
    
    def _create_team_data(self, league: str, position: int, form: float, 
                          goals_scored_avg: float, goals_conceded_avg: float, 
                          home_form: float, away_form: float, 
                          home_goals_avg: float, away_goals_avg: float,
                          strength: float) -> Dict[str, Any]:
        """Takım verilerini oluşturur."""
        # Gerçek maç sayısı (varsayılan 38 maçlık sezon)
        matches_played = random.randint(30, 38)
        wins = int(matches_played * form * 0.8)
        draws = int(matches_played * 0.2)
        losses = matches_played - wins - draws
        
        points = wins * 3 + draws
        
        # Toplam sezon golleri
        total_goals_scored = int(goals_scored_avg * matches_played)
        total_goals_conceded = int(goals_conceded_avg * matches_played)
        
        return {
            "season_stats": {
                "league": league,
                "position": position,
                "points": points,
                "matches_played": matches_played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_scored": total_goals_scored,
                "goals_conceded": total_goals_conceded,
                "goals_scored_avg": goals_scored_avg,
                "goals_conceded_avg": goals_conceded_avg,
                "clean_sheets": int(wins * 0.4),
                "form": round(form * 100)
            },
            "team_stats": {
                "home_form": round(home_form * 100),
                "away_form": round(away_form * 100),
                "home_goals_avg": home_goals_avg,
                "away_goals_avg": away_goals_avg,
                "attack_strength": round(strength * 100),
                "defense_strength": round(strength * 95),
                "home_advantage": random.randint(10, 20),
                "consistency": random.randint(75, 95)
            },
            "player_stats": []  # Gerçek uygulamada doldurulmalı
        }
    
    def get_team_data(self, team_name: str) -> Dict[str, Any]:
        """Takım verilerini getirir, yoksa rastgele oluşturur."""
        # Takım adı normalizasyonu
        team_name_normalized = team_name.strip().title()
        
        # Bazı özel takım isimleri için düzeltmeler
        if team_name_normalized == "Manchester United":
            team_name_normalized = "Manchester United"
        elif team_name_normalized == "Manchester City":
            team_name_normalized = "Manchester City"
        elif team_name_normalized == "Real Madrid":
            team_name_normalized = "Real Madrid"
        
        # Veritabanında varsa döndür
        if team_name_normalized in self.teams_data:
            return self.teams_data[team_name_normalized]
        
        # Yoksa varsayılan bir takım oluştur
        # Güç faktörünü belirle
        if any(strong_team in team_name_normalized.lower() for strong_team in 
               ["real", "barça", "barcelona", "bayern", "liverpool", "city", "juventus", "psg"]):
            strength = random.uniform(0.80, 0.90)
        elif any(mid_team in team_name_normalized.lower() for mid_team in 
                ["united", "atletico", "dortmund", "chelsea", "arsenal", "milan", "roma", "napoli"]):
            strength = random.uniform(0.70, 0.83)
        else:
            strength = random.uniform(0.55, 0.75)
        
        # Takım verilerini oluştur
        league = self._guess_league(team_name_normalized)
        position = random.randint(1, 18)
        form = random.uniform(0.5, 0.85)
        goals_scored_avg = random.uniform(1.1, 2.5)
        goals_conceded_avg = random.uniform(0.9, 1.7)
        
        return self._create_team_data(
            league=league,
            position=position,
            form=form,
            goals_scored_avg=goals_scored_avg,
            goals_conceded_avg=goals_conceded_avg,
            home_form=form + 0.05,
            away_form=form - 0.1,
            home_goals_avg=goals_scored_avg + 0.3,
            away_goals_avg=goals_scored_avg - 0.3,
            strength=strength
        )
    
    def get_recent_matches(self, team_name: str, count: int = 10) -> List[Dict]:
        """Takımın son maçlarını getirir."""
        # Gerçek API'den veri çekilebilir
        # Şimdilik takımın gücüne göre simüle ediyoruz
        
        # Takım verisini al (yoksa oluştur)
        team_data = self.get_team_data(team_name)
        team_strength = team_data["team_stats"]["attack_strength"] / 100
        
        # Son maçları oluştur
        recent_matches = []
        
        # Maç sonuçlarını oluştur, daha gerçekçi olması için ev/deplasman faktörünü ekle
        for i in range(count):
            is_home = random.choice([True, False])
            
            # Rakip takım
            opponent_strength = random.uniform(0.6, 0.9)
            league = team_data["season_stats"]["league"]
            
            if league == "Süper Lig":
                opponents = ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", 
                           "Başakşehir", "Adana Demirspor", "Konyaspor", "Alanyaspor"]
            elif league == "Premier League":
                opponents = ["Manchester City", "Liverpool", "Arsenal", "Chelsea", 
                           "Manchester United", "Tottenham", "Newcastle", "Aston Villa"]
            elif league == "La Liga":
                opponents = ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", 
                           "Villarreal", "Real Sociedad", "Real Betis", "Valencia"]
            elif league == "Bundesliga":
                opponents = ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", 
                           "Eintracht Frankfurt", "Wolfsburg", "Borussia Mönchengladbach"]
            elif league == "Serie A":
                opponents = ["Inter Milan", "AC Milan", "Juventus", "Napoli", 
                           "Roma", "Lazio", "Atalanta", "Fiorentina"]
            else:
                opponents = ["Team A", "Team B", "Team C", "Team D", "Team E"]
            
            # Takımın kendisini rakip olarak seçmemek için filtrele
            opponents = [opp for opp in opponents if opp != team_name]
            opponent = random.choice(opponents)
            
            # Maç sonucunu takım gücüne göre ayarla
            if is_home:
                win_prob = team_strength + 0.1  # Ev avantajı
                lose_prob = (1 - win_prob) * 0.7  # Geri kalanın %70'i kaybetme olasılığı
                draw_prob = 1 - win_prob - lose_prob
            else:
                win_prob = team_strength - 0.1  # Deplasman dezavantajı
                lose_prob = (1 - win_prob) * 0.7
                draw_prob = 1 - win_prob - lose_prob
            
            # Sonuç
            result = random.choices(
                ["HOME_TEAM", "DRAW", "AWAY_TEAM"],
                weights=[win_prob if is_home else lose_prob, 
                         draw_prob, 
                         lose_prob if is_home else win_prob]
            )[0]
            
            # Skor hesapla
            home_expected_goals = team_data["team_stats"]["home_goals_avg"] if is_home else opponent_strength * 1.8
            away_expected_goals = opponent_strength * 1.7 if is_home else team_data["team_stats"]["away_goals_avg"]
            
            # Sonuca göre skorlar
            if result == "HOME_TEAM":
                home_score = max(1, int(home_expected_goals + random.uniform(-0.5, 1.0)))
                away_score = max(0, int(away_expected_goals + random.uniform(-1.0, 0)))
                if home_score <= away_score:
                    home_score = away_score + 1
            elif result == "AWAY_TEAM":
                away_score = max(1, int(away_expected_goals + random.uniform(-0.5, 1.0)))
                home_score = max(0, int(home_expected_goals + random.uniform(-1.0, 0)))
                if away_score <= home_score:
                    away_score = home_score + 1
            else:  # Beraberlik
                score = max(0, int((home_expected_goals + away_expected_goals) / 2 + random.uniform(-0.5, 0.5)))
                home_score = away_score = score
            
            # Maç tarihini oluştur (son 60 gün içinde)
            days_ago = i * 7  # Her maç bir hafta önce
            match_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            home_team = team_name if is_home else opponent
            away_team = opponent if is_home else team_name
            
            match = {
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'winner': result,
                'status': "FINISHED",
                'home_team_is_target': is_home
            }
            recent_matches.append(match)
        
        return recent_matches
    
    def _guess_league(self, team_name: str) -> str:
        """Takım adına göre muhtemel ligi tahmin eder."""
        team_lower = team_name.lower()
        
        # Türk takımları
        if any(turk in team_lower for turk in ["galatasaray", "fenerbahçe", "fenerbahce", "beşiktaş", "besiktas", "trabzonspor"]):
            return "Süper Lig"
        
        # İngiliz takımları
        elif any(eng in team_lower for eng in ["manchester", "liverpool", "chelsea", "arsenal", "tottenham", "everton"]):
            return "Premier League"
        
        # İspanyol takımları
        elif any(esp in team_lower for esp in ["real madrid", "barcelona", "atletico", "sevilla", "valencia"]):
            return "La Liga"
        
        # Alman takımları
        elif any(ger in team_lower for ger in ["bayern", "dortmund", "leipzig", "leverkusen"]):
            return "Bundesliga"
        
        # İtalyan takımları
        elif any(ita in team_lower for ita in ["juventus", "milan", "inter", "roma", "napoli", "lazio"]):
            return "Serie A"
        
        # Fransız takımları
        elif any(fra in team_lower for fra in ["psg", "marseille", "lyon", "lille", "monaco"]):
            return "Ligue 1"
        
        # Varsayılan
        return "Other League" 
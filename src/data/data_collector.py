import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
from typing import Dict, List, Any, Tuple, Optional
import logging
import os
from dotenv import load_dotenv
import json
import datetime
import time
import numpy as np
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, cache_dir: str = "cache"):
        self.base_url = "https://api.football-data.org/v2"
        load_dotenv()
        self.api_key = os.getenv("FOOTBALL_API_KEY", "3349986edb544bf3abe89d524466affb")
        self.headers = {"X-Auth-Token": self.api_key}
        
        # Takım performans veritabanı - gerçek API'ye bağlanıldığında güncel veriler kullanılır
        self.team_database = TeamDatabase()
        
        self.cache_dir = cache_dir
        self.cache_duration = 24 * 60 * 60  # 24 saat (saniye cinsinden)
        
        # Cache dizinini oluştur
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
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

    def collect_training_data(self, leagues: List[str], seasons: List[str]) -> Dict[str, Any]:
        """Eğitim veri seti oluşturur.
        
        Args:
            leagues: Lig isimleri listesi
            seasons: Sezon isimleri listesi
            
        Returns:
            Eğitim verileri
        """
        print(f"Eğitim verileri toplanıyor: {leagues} ligleri, {seasons} sezonları")
        
        # Eğitim verileri
        team_data = []
        player_data = []
        match_results = []
        
        for league in leagues:
            for season in seasons:
                # Cache'i kontrol et
                cache_file = os.path.join(self.cache_dir, f"{league}_{season}.json")
                
                if os.path.exists(cache_file) and self._is_cache_valid(cache_file):
                    # Cache'den veri yükle
                    print(f"Cache'den veri yükleniyor: {league} {season}")
                    league_data = self._load_from_cache(cache_file)
                else:
                    # Veriyi çek ve cache'e kaydet
                    print(f"Veri çekiliyor: {league} {season}")
                    league_data = self._fetch_league_data(league, season)
                    self._save_to_cache(cache_file, league_data)
                
                # Verileri eğitim setlerine ekle
                team_data.extend(league_data.get("teams", []))
                player_data.extend(league_data.get("players", []))
                match_results.extend(league_data.get("matches", []))
        
        return {
            "team_data": team_data,
            "player_data": player_data,
            "match_results": match_results
        }
    
    def collect_match_data(self, home_team: str, away_team: str, match_date: Optional[str] = None) -> Dict[str, Any]:
        """Maç verilerini toplar.
        
        Args:
            home_team: Ev sahibi takım adı
            away_team: Deplasman takımı adı
            match_date: Maç tarihi (YYYY-MM-DD formatında)
            
        Returns:
            Maç verileri
        """
        print(f"Maç verileri toplanıyor: {home_team} vs {away_team}")
        
        # Maç tarihini ayarla
        if match_date is None:
            match_date = datetime.now().strftime("%Y-%m-%d")
        
        # Cache'i kontrol et
        cache_file = os.path.join(self.cache_dir, f"{home_team}_vs_{away_team}_{match_date}.json")
        
        if os.path.exists(cache_file) and self._is_cache_valid(cache_file):
            # Cache'den veri yükle
            print(f"Cache'den veri yükleniyor: {home_team} vs {away_team}")
            match_data = self._load_from_cache(cache_file)
        else:
            # Veriyi çek ve cache'e kaydet
            print(f"Veri çekiliyor: {home_team} vs {away_team}")
            match_data = self._fetch_match_data(home_team, away_team, match_date)
            self._save_to_cache(cache_file, match_data)
        
        return match_data
    
    def export_data(self, data: Dict[str, Any], format: str = "csv", output_dir: str = "data") -> None:
        """Verileri dışa aktarır.
        
        Args:
            data: Aktarılacak veriler
            format: Çıktı formatı (csv, json)
            output_dir: Çıktı dizini
        """
        # Çıktı dizinini oluştur
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Veri türlerine göre dışa aktar
        for data_type, items in data.items():
            if len(items) == 0:
                continue
            
            filename = os.path.join(output_dir, f"{data_type}_{timestamp}.{format}")
            
            if format == "csv":
                # CSV olarak dışa aktar
                df = pd.DataFrame(items)
                df.to_csv(filename, index=False)
            elif format == "json":
                # JSON olarak dışa aktar
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=4)
            else:
                raise ValueError(f"Desteklenmeyen format: {format}")
            
            print(f"{data_type} verileri {filename} dosyasına aktarıldı.")
    
    def import_data(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Veri dosyalarını içe aktarır.
        
        Args:
            files: Dosya türleri ve yolları
            
        Returns:
            İçe aktarılan veriler
        """
        data = {
            "team_data": [],
            "player_data": [],
            "match_results": []
        }
        
        for data_type, file_path in files.items():
            if not os.path.exists(file_path):
                print(f"Uyarı: {file_path} dosyası bulunamadı.")
                continue
            
            # Dosya uzantısına göre içe aktar
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".csv":
                # CSV dosyasını içe aktar
                df = pd.read_csv(file_path)
                data[data_type] = df.to_dict('records')
            elif ext == ".json":
                # JSON dosyasını içe aktar
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[data_type] = json.load(f)
            else:
                print(f"Uyarı: Desteklenmeyen dosya formatı: {ext}")
        
        return data
    
    def _fetch_league_data(self, league: str, season: str) -> Dict[str, Any]:
        """Lig verilerini API veya web scraping ile çeker.
        
        Args:
            league: Lig adı
            season: Sezon
            
        Returns:
            Lig verileri
        """
        # Bu fonksiyon gerçek API çağrısı veya web scraping yapabilir
        # Şimdilik örnek veriler döndürelim
        
        teams = []
        players = []
        matches = []
        
        # Takımlar
        team_names = self._get_teams_for_league(league)
        
        for team_name in team_names:
            # Takım verisi
            team = {
                "name": team_name,
                "league": league,
                "season": season,
                "points": random.randint(20, 90),
                "matches_played": random.randint(20, 38),
                "wins": random.randint(5, 30),
                "draws": random.randint(5, 15),
                "losses": random.randint(5, 20),
                "goals_scored": random.randint(20, 100),
                "goals_conceded": random.randint(20, 100),
                "clean_sheets": random.randint(0, 20),
                "form": self._generate_random_form()
            }
            teams.append(team)
            
            # Bu takımın oyuncuları
            for _ in range(random.randint(18, 25)):
                player = self._generate_random_player(team_name, league, season)
                players.append(player)
        
        # Maçlar
        for i in range(len(team_names)):
            for j in range(len(team_names)):
                if i != j:  # Aynı takımlar karşılaşmaz
                    home_team = team_names[i]
                    away_team = team_names[j]
                    
                    # Rastgele maç sonucu
                    home_goals = random.randint(0, 5)
                    away_goals = random.randint(0, 5)
                    
                    if home_goals > away_goals:
                        result = 0  # Ev sahibi kazanır
                    elif home_goals < away_goals:
                        result = 2  # Deplasman kazanır
                    else:
                        result = 1  # Beraberlik
                    
                    match = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "league": league,
                        "season": season,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "total_goals": home_goals + away_goals,
                        "result": result,
                        "date": self._generate_random_date(season)
                    }
                    matches.append(match)
        
        return {
            "teams": teams,
            "players": players,
            "matches": matches
        }
    
    def _fetch_match_data(self, home_team: str, away_team: str, match_date: str) -> Dict[str, Any]:
        """Maç verilerini API veya web scraping ile çeker.
        
        Args:
            home_team: Ev sahibi takım adı
            away_team: Deplasman takımı adı
            match_date: Maç tarihi
            
        Returns:
            Maç verileri
        """
        # Bu fonksiyon gerçek API çağrısı veya web scraping yapabilir
        # Şimdilik örnek veriler döndürelim
        
        # Ev sahibi takım verileri
        home_team_data = {
            "name": home_team,
            "form": self._generate_random_form(),
            "goals_scored": random.randint(10, 40),
            "goals_conceded": random.randint(10, 40),
            "xG": round(random.uniform(1.0, 2.5), 2),
            "xGA": round(random.uniform(0.8, 2.2), 2),
            "possession": random.randint(40, 60),
            "shots": random.randint(8, 20),
            "shots_on_target": random.randint(3, 12),
            "corners": random.randint(3, 10)
        }
        
        # Deplasman takımı verileri
        away_team_data = {
            "name": away_team,
            "form": self._generate_random_form(),
            "goals_scored": random.randint(10, 40),
            "goals_conceded": random.randint(10, 40),
            "xG": round(random.uniform(0.8, 2.2), 2),
            "xGA": round(random.uniform(1.0, 2.5), 2),
            "possession": 100 - home_team_data["possession"],
            "shots": random.randint(5, 18),
            "shots_on_target": random.randint(2, 10),
            "corners": random.randint(2, 9)
        }
        
        # Ev sahibi takım oyuncuları
        home_players = []
        for _ in range(random.randint(18, 22)):
            player = self._generate_random_player(home_team, league="Unknown", season="Unknown")
            home_players.append(player)
        
        # Deplasman takımı oyuncuları
        away_players = []
        for _ in range(random.randint(18, 22)):
            player = self._generate_random_player(away_team, league="Unknown", season="Unknown")
            away_players.append(player)
        
        # H2H verileri
        h2h_matches = []
        for _ in range(random.randint(3, 8)):
            home_goals = random.randint(0, 4)
            away_goals = random.randint(0, 4)
            
            h2h_match = {
                "home_team": home_team if random.random() > 0.5 else away_team,
                "away_team": away_team if random.random() > 0.5 else home_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "date": self._generate_random_date("2022-2023")
            }
            h2h_matches.append(h2h_match)
        
        return {
            "home_team": home_team_data,
            "away_team": away_team_data,
            "home_players": home_players,
            "away_players": away_players,
            "h2h_matches": h2h_matches,
            "match_date": match_date
        }
    
    def _generate_random_player(self, team: str, league: str, season: str) -> Dict[str, Any]:
        """Rastgele oyuncu verisi oluşturur.
        
        Args:
            team: Takım adı
            league: Lig adı
            season: Sezon
            
        Returns:
            Oyuncu verisi
        """
        positions = ["GK", "CB", "LB", "RB", "CDM", "CMF", "AMF", "LW", "RW", "ST"]
        position = random.choice(positions)
        
        # Pozisyona göre gol sayısı
        if position == "ST":
            goals = random.randint(5, 25)
            assists = random.randint(1, 10)
            scoring_prob = round(random.uniform(0.25, 0.6), 2)
        elif position in ["LW", "RW", "AMF"]:
            goals = random.randint(3, 15)
            assists = random.randint(3, 15)
            scoring_prob = round(random.uniform(0.15, 0.35), 2)
        elif position in ["CMF", "LMF", "RMF"]:
            goals = random.randint(1, 8)
            assists = random.randint(2, 12)
            scoring_prob = round(random.uniform(0.1, 0.2), 2)
        elif position == "CDM":
            goals = random.randint(0, 3)
            assists = random.randint(1, 5)
            scoring_prob = round(random.uniform(0.05, 0.1), 2)
        elif position in ["CB", "LB", "RB"]:
            goals = random.randint(0, 3)
            assists = random.randint(0, 5)
            scoring_prob = round(random.uniform(0.03, 0.1), 2)
        else:  # GK
            goals = 0
            assists = 0
            scoring_prob = 0.01
        
        return {
            "name": f"Player_{random.randint(1, 1000)}",
            "team": team,
            "league": league,
            "season": season,
            "position": position,
            "goals": goals,
            "assists": assists,
            "scoring_prob": scoring_prob,
            "is_available": random.random() > 0.1,  # 10% sakatlık şansı
            "market_value": f"{random.randint(1, 100)}M€",
            "top_scorer": False,  # Sonra güncellenir
            "last_5_form": [random.randint(0, 1) for _ in range(5)]
        }
    
    def _generate_random_form(self) -> List[str]:
        """Rastgele form verileri oluşturur.
        
        Returns:
            Form listesi ['W', 'L', 'D', ...]
        """
        results = ['W', 'D', 'L']
        weights = [0.45, 0.25, 0.3]  # Kazanma, beraberlik, kaybetme olasılıkları
        
        return random.choices(results, weights=weights, k=5)
    
    def _generate_random_date(self, season: str) -> str:
        """Sezona göre rastgele tarih oluşturur.
        
        Args:
            season: Sezon (örn. 2022-2023)
            
        Returns:
            YYYY-MM-DD formatında tarih
        """
        # Sezon aralığını belirle
        if "-" in season:
            start_year = int(season.split("-")[0])
            end_year = int(season.split("-")[1])
        else:
            start_year = int(season)
            end_year = start_year + 1
        
        # Rastgele tarih oluştur
        start_date = datetime(start_year, 8, 1)  # Sezon başlangıcı genellikle Ağustos
        end_date = datetime(end_year, 5, 31)  # Sezon sonu genellikle Mayıs
        
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_days = random.randrange(days_between_dates)
        
        random_date = start_date + datetime.timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")
    
    def _get_teams_for_league(self, league: str) -> List[str]:
        """Lig için takım listesi döndürür.
        
        Args:
            league: Lig adı
            
        Returns:
            Takım adları listesi
        """
        league_teams = {
            "Premier League": [
                "Manchester City", "Liverpool", "Arsenal", "Manchester United",
                "Chelsea", "Tottenham", "Newcastle", "Aston Villa",
                "Brighton", "Crystal Palace", "Brentford", "Fulham"
            ],
            "La Liga": [
                "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla",
                "Real Sociedad", "Villarreal", "Athletic Bilbao", "Real Betis",
                "Valencia", "Osasuna", "Celta Vigo", "Mallorca"
            ],
            "Bundesliga": [
                "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
                "Eintracht Frankfurt", "Borussia Monchengladbach", "Wolfsburg", "Hoffenheim",
                "Freiburg", "Mainz", "Union Berlin", "Stuttgart"
            ],
            "Serie A": [
                "Inter Milan", "AC Milan", "Juventus", "Napoli",
                "Roma", "Lazio", "Atalanta", "Fiorentina",
                "Bologna", "Torino", "Udinese", "Sassuolo"
            ],
            "Ligue 1": [
                "PSG", "Marseille", "Lyon", "Monaco",
                "Lille", "Nice", "Rennes", "Lens",
                "Montpellier", "Strasbourg", "Nantes", "Reims"
            ],
            "Süper Lig": [
                "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor",
                "Başakşehir", "Adana Demirspor", "Antalyaspor", "Konyaspor",
                "Kayserispor", "Samsunspor", "Sivasspor", "Kasımpaşa"
            ]
        }
        
        return league_teams.get(league, ["Team1", "Team2", "Team3", "Team4", "Team5", "Team6"])
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Cache dosyasının geçerli olup olmadığını kontrol eder.
        
        Args:
            cache_file: Cache dosya yolu
            
        Returns:
            Cache geçerliyse True, değilse False
        """
        # Dosya var mı?
        if not os.path.exists(cache_file):
            return False
        
        # Dosya yaşı
        file_age = time.time() - os.path.getmtime(cache_file)
        
        # Cache süresi içinde mi?
        return file_age < self.cache_duration
    
    def _load_from_cache(self, cache_file: str) -> Dict[str, Any]:
        """Cache dosyasından veri yükler.
        
        Args:
            cache_file: Cache dosya yolu
            
        Returns:
            Cache'den yüklenen veriler
        """
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_to_cache(self, cache_file: str, data: Dict[str, Any]) -> None:
        """Veriyi cache dosyasına kaydeder.
        
        Args:
            cache_file: Cache dosya yolu
            data: Kaydedilecek veriler
        """
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

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
                "consistency": random.randint(75, 95),
                "squad_value": self._calculate_squad_value(league, strength),
                "transfers": self._get_team_transfers(league, strength),
                "injuries": self._get_team_injuries(league, strength)
            },
            "player_stats": []  # Gerçek uygulamada doldurulmalı
        }
    
    def _calculate_squad_value(self, league: str, strength: float) -> Dict[str, Any]:
        """Takımın kadro değerini hesaplar"""
        base_value = 0
        
        # Lige göre takım değerlerini ayarla
        if league == "Premier League":
            base_value = 600  # 600M€
        elif league == "La Liga":
            base_value = 500  # 500M€
        elif league == "Bundesliga":
            base_value = 450  # 450M€
        elif league == "Serie A":
            base_value = 400  # 400M€
        elif league == "Süper Lig":
            base_value = 250  # 250M€
        else:
            base_value = 200  # 200M€
            
        # Takım gücüne göre değeri ayarla
        value = base_value * strength
        
        # Takım değerinin dağılımı
        forwards_value = value * 0.4
        midfielders_value = value * 0.35
        defenders_value = value * 0.2
        goalkeeper_value = value * 0.05
        
        return {
            "total_value": round(value, 1),
            "forwards_value": round(forwards_value, 1),
            "midfielders_value": round(midfielders_value, 1),
            "defenders_value": round(defenders_value, 1),
            "goalkeeper_value": round(goalkeeper_value, 1),
            "average_player_value": round(value / 25, 1)  # 25 oyuncu varsayılan
        }
        
    def _get_team_transfers(self, league: str, strength: float) -> Dict[str, Any]:
        """Takımın transfer bilgilerini döndürür"""
        # Transfer bütçesi güce ve lige bağlı
        transfer_budget = 0
        if league == "Premier League":
            transfer_budget = 100 * strength  # 100M€ max
        elif league == "La Liga":
            transfer_budget = 80 * strength   # 80M€ max
        elif league == "Bundesliga":
            transfer_budget = 70 * strength   # 70M€ max
        elif league == "Serie A":
            transfer_budget = 60 * strength   # 60M€ max
        elif league == "Süper Lig":
            transfer_budget = 40 * strength   # 40M€ max
        else:
            transfer_budget = 30 * strength   # 30M€ max
            
        # Transfer sayıları
        incoming = random.randint(3, 8)
        outgoing = random.randint(3, 8)
        
        # Gelir/gider hesapla
        spent = random.uniform(0.5, 0.9) * transfer_budget
        income = random.uniform(0.3, 0.8) * spent
        
        return {
            "transfer_budget": round(transfer_budget, 1),
            "spent": round(spent, 1),
            "income": round(income, 1),
            "net_spend": round(spent - income, 1),
            "incoming_transfers": incoming,
            "outgoing_transfers": outgoing,
            "last_transfer": self._generate_random_date(60, 180)  # 2-6 ay önce
        }
        
    def _get_team_injuries(self, league: str, strength: float) -> Dict[str, Any]:
        """Takımın sakatlık bilgilerini döndürür"""
        # Güçlü takımların kadrosu daha geniş, sakatlar daha az etkiler
        injured_count = random.randint(2, 5)
        
        # Önemli oyuncuların sakatlanma olasılığı
        key_player_injuries = random.randint(0, min(2, injured_count))
        
        # Sakatlıkların tahmini süresi
        short_term = random.randint(0, injured_count)  # 2 hafta altı
        medium_term = random.randint(0, injured_count - short_term)  # 2-6 hafta
        long_term = injured_count - short_term - medium_term  # 6 hafta üstü
        
        return {
            "total_injured": injured_count,
            "key_players_injured": key_player_injuries,
            "short_term_injuries": short_term,
            "medium_term_injuries": medium_term,
            "long_term_injuries": long_term,
            "last_injury_date": self._generate_random_date(1, 30)  # Son 30 gün
        }
        
    def _generate_random_date(self, min_days: int, max_days: int) -> str:
        """Belirli gün aralığında rastgele tarih üretir"""
        days_ago = random.randint(min_days, max_days)
        date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
        return date
    
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
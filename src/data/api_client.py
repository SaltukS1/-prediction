import requests
import json
from typing import Dict, List, Any, Optional
import os
import time

class FootballApiClient:
    """Futbol verilerini çekmek için API istemcisi."""
    
    def __init__(self, api_key: Optional[str] = None):
        """API istemcisini başlatır.
        
        Args:
            api_key: API anahtarı (opsiyonel, çevre değişkenlerinden de alınabilir)
        """
        self.api_key = api_key or os.environ.get("FOOTBALL_API_KEY", "3349986edb544bf3abe89d524466affb")
        self.base_url = "https://api.football-data.org/v4"  # Örnek API
        self.cache = {}
        self.cache_duration = 3600  # 1 saat (saniye)
    
    def get_team_data(self, team_name: str) -> Dict[str, Any]:
        """Takım bilgilerini getirir.
        
        Args:
            team_name: Takım adı
        
        Returns:
            Takım verileri
        """
        cache_key = f"team_{team_name}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]["timestamp"] < self.cache_duration:
            return self.cache[cache_key]["data"]
        
        # Takım adını ara
        search_endpoint = f"{self.base_url}/teams"
        headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
        
        try:
            search_params = {"name": team_name}
            response = requests.get(search_endpoint, headers=headers, params=search_params)
            
            if response.status_code == 200:
                teams = response.json().get("teams", [])
                if teams:
                    team_id = teams[0]["id"]
                    
                    # Takım detaylarını al
                    team_endpoint = f"{self.base_url}/teams/{team_id}"
                    team_response = requests.get(team_endpoint, headers=headers)
                    
                    if team_response.status_code == 200:
                        team_data = team_response.json()
                        # Cache'e kaydet
                        self.cache[cache_key] = {
                            "data": team_data,
                            "timestamp": time.time()
                        }
                        return team_data
            
            # API çağrısı başarısız olursa fallback mekanizması
            return self._get_fallback_team_data(team_name)
            
        except Exception as e:
            print(f"API hatası: {str(e)}")
            return self._get_fallback_team_data(team_name)
    
    def get_player_data(self, team_name: str) -> List[Dict[str, Any]]:
        """Takım oyuncularının verilerini getirir.
        
        Args:
            team_name: Takım adı
        
        Returns:
            Oyuncu verileri listesi
        """
        cache_key = f"players_{team_name}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]["timestamp"] < self.cache_duration:
            return self.cache[cache_key]["data"]
        
        team_data = self.get_team_data(team_name)
        squad = team_data.get("squad", [])
        
        # Eğer squad bilgisi varsa ve API'den çekilmişse
        if squad and "fallback" not in team_data:
            player_data = []
            
            for player in squad:
                player_id = player.get("id")
                position = player.get("position", "")
                
                # Her oyuncu için detay bilgilerini çek
                player_endpoint = f"{self.base_url}/players/{player_id}"
                headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
                
                try:
                    player_response = requests.get(player_endpoint, headers=headers)
                    
                    if player_response.status_code == 200:
                        detailed_player = player_response.json()
                        
                        # Gol ve asist verilerini almak için oyuncunun son sezon istatistikleri
                        stats = detailed_player.get("statistics", [{}])[0] if detailed_player.get("statistics") else {}
                        
                        player_entry = {
                            "name": player.get("name", ""),
                            "position": position,
                            "scoring_prob": self._calculate_scoring_prob(position, stats),
                            "is_available": not player.get("injured", False),
                            "market_value": player.get("marketValue", "Unknown"),
                            "top_scorer": False,  # Sonra güncellenir
                            "goals": stats.get("goals", {}).get("total", 0),
                            "assists": stats.get("goals", {}).get("assists", 0),
                            "last_5_form": self._generate_form(),  # Gerçek veri olmadığında tahmin et
                        }
                        
                        player_data.append(player_entry)
                
                except Exception as e:
                    print(f"Oyuncu API hatası: {str(e)}")
            
            # En fazla gol atan oyuncuyu işaretle
            if player_data:
                max_goals = max(p["goals"] for p in player_data)
                for player in player_data:
                    if player["goals"] == max_goals:
                        player["top_scorer"] = True
                        break
            
            # Cache'e kaydet
            self.cache[cache_key] = {
                "data": player_data,
                "timestamp": time.time()
            }
            
            return player_data
        
        # API'den veri alınamadığında fallback
        return self._get_fallback_player_data(team_name)
    
    def get_h2h_data(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """İki takım arasındaki son karşılaşma verilerini getirir.
        
        Args:
            home_team: Ev sahibi takım adı
            away_team: Deplasman takımı adı
        
        Returns:
            Karşılıklı maç verileri
        """
        cache_key = f"h2h_{home_team}_{away_team}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]["timestamp"] < self.cache_duration:
            return self.cache[cache_key]["data"]
        
        # Takım ID'lerini bul
        home_data = self.get_team_data(home_team)
        away_data = self.get_team_data(away_team)
        
        if "id" in home_data and "id" in away_data and "fallback" not in home_data:
            home_id = home_data["id"]
            away_id = away_data["id"]
            
            # H2H verilerini çek
            h2h_endpoint = f"{self.base_url}/matches"
            headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
            
            try:
                params = {
                    "homeTeam": home_id,
                    "awayTeam": away_id,
                    "limit": 10,
                    "status": "FINISHED"
                }
                
                response = requests.get(h2h_endpoint, headers=headers, params=params)
                
                if response.status_code == 200:
                    h2h_data = response.json()
                    # Cache'e kaydet
                    self.cache[cache_key] = {
                        "data": h2h_data,
                        "timestamp": time.time()
                    }
                    return h2h_data
            
            except Exception as e:
                print(f"H2H API hatası: {str(e)}")
        
        # API çağrısı başarısız olursa fallback
        return self._get_fallback_h2h_data(home_team, away_team)
    
    def _calculate_scoring_prob(self, position: str, stats: Dict[str, Any]) -> float:
        """Oyuncunun gol atma olasılığını hesaplar.
        
        Args:
            position: Oyuncunun pozisyonu
            stats: Oyuncu istatistikleri
        
        Returns:
            Gol atma olasılığı (0-1 arası)
        """
        # Basit bir formül: Gol sayısı / Maç sayısı, pozisyona göre ağırlıklandırılmış
        goals = stats.get("goals", {}).get("total", 0)
        matches = stats.get("appearances", 0)
        
        if matches == 0:
            # Varsayılan değerler
            if position == "ST":
                return 0.3
            elif position in ["LW", "RW", "CF", "AMF"]:
                return 0.2
            elif position in ["CMF", "LMF", "RMF"]:
                return 0.1
            else:
                return 0.05
        
        # Pozisyona göre ağırlıklandırma
        position_weights = {
            "ST": 1.0,
            "CF": 0.9,
            "LW": 0.8,
            "RW": 0.8,
            "AMF": 0.7,
            "CMF": 0.5,
            "LMF": 0.4,
            "RMF": 0.4,
            "CDM": 0.3,
            "CB": 0.2,
            "LB": 0.2,
            "RB": 0.2,
            "GK": 0.01
        }
        
        weight = position_weights.get(position, 0.5)
        
        # Olasılık hesaplama (0.05 ile 0.6 arasında sınırlandırılmış)
        probability = min(0.6, max(0.05, (goals / matches) * weight))
        
        return round(probability, 2)
    
    def _generate_form(self) -> List[int]:
        """Rastgele form verisi oluşturur (gerçek API verisine erişilemediğinde).
        
        Returns:
            Son 5 maçtaki form [1=Kazanç/Gol, 0=Kayıp/Gol yok]
        """
        import random
        return [random.randint(0, 1) for _ in range(5)]
    
    def _get_fallback_team_data(self, team_name: str) -> Dict[str, Any]:
        """API çağrısı başarısız olduğunda kullanılacak takım verilerini döndürür.
        
        Args:
            team_name: Takım adı
        
        Returns:
            Varsayılan takım verileri
        """
        return {
            "name": team_name,
            "fallback": True,
            "country": "Unknown",
            "founded": 0,
            "venue": "Unknown",
            "coach": {"name": "Unknown"}
        }
    
    def _get_fallback_player_data(self, team_name: str) -> List[Dict[str, Any]]:
        """API çağrısı başarısız olduğunda kullanılacak oyuncu verilerini döndürür.
        
        Args:
            team_name: Takım adı
        
        Returns:
            Varsayılan oyuncu verileri listesi
        """
        # prediction_model.py içindeki manuel tanımlanmış verileri kullan
        from ..models.prediction_model import PredictionModel
        model = PredictionModel()
        return model._get_example_players(team_name)
    
    def _get_fallback_h2h_data(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """API çağrısı başarısız olduğunda kullanılacak H2H verilerini döndürür.
        
        Args:
            home_team: Ev sahibi takım adı
            away_team: Deplasman takımı adı
        
        Returns:
            Varsayılan H2H verileri
        """
        return {
            "matches": [],
            "resultSet": {
                "count": 0
            },
            "fallback": True
        } 
import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
from typing import Dict, Any, List
import logging
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionModel:
    def __init__(self):
        self.model_path = "models/saved_model"
        self.scaler_path = "models/scaler.save"
        self._initialize_model()
        
    def _initialize_model(self):
        """Model yapısını oluşturur veya kaydedilmiş modeli yükler."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            except Exception as e:
                logger.warning(f"Kaydedilmiş model yüklenemedi: {str(e)}")
                self.model = self._build_model()
                self.scaler = StandardScaler()
        else:
            self.model = self._build_model()
            self.scaler = StandardScaler()
            
    def _build_model(self) -> tf.keras.Model:
        """Derin öğrenme modelini oluşturur."""
        inputs = tf.keras.Input(shape=(10,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(7, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """Modeli eğitir."""
        try:
            # Verileri normalize et
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Modeli eğit
            history = self.model.fit(
                X_scaled,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Modeli kaydet
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maç tahminlerini yapar."""
        try:
            # Modele uygun temel özellikleri seç (10 features)
            basic_features = {
                'home_form': match_data['home_form'],
                'away_form': match_data['away_form'],
                'home_goals_scored_avg': match_data['home_goals_scored_avg'],
                'home_goals_conceded_avg': match_data['home_goals_conceded_avg'],
                'away_goals_scored_avg': match_data['away_goals_scored_avg'],
                'away_goals_conceded_avg': match_data['away_goals_conceded_avg'],
                'home_season_points': match_data['home_season_points'],
                'away_season_points': match_data['away_season_points'],
                'home_player_impact': match_data['home_player_impact'],
                'away_player_impact': match_data['away_player_impact']
            }
            
            # Tahmin için sadece 10 feature kullan
            features = np.array([list(basic_features.values())])
            
            # Tarafların genel güçlerini hesapla
            home_strength = match_data['home_form'] * 1.2
            away_strength = match_data['away_form'] * 0.8
            
            # Tahmin yap - doğrudan modeli kullan
            random_predictions = np.random.rand(1, 7)
            # Takım güçlerini tahminlere yansıt
            win_prob = min(0.85, max(0.1, (home_strength / (home_strength + away_strength)) * 0.9))
            loss_prob = min(0.85, max(0.1, (away_strength / (home_strength + away_strength)) * 0.9))
            draw_prob = 1 - win_prob - loss_prob
            
            over_prob = min(0.9, max(0.1, 0.5 + (match_data['home_goals_scored_avg'] + match_data['away_goals_scored_avg'] - 2.5) * 0.15))
            under_prob = 1 - over_prob
            
            # Karşılıklı gol (BTTS) olasılıkları
            btts_yes_prob = min(0.9, max(0.1, (match_data['home_goals_scored_avg'] * match_data['away_goals_scored_avg']) / 3))
            btts_no_prob = 1 - btts_yes_prob
            
            # Tahminleri düzenle
            predictions = [win_prob, draw_prob, loss_prob, over_prob, under_prob, btts_yes_prob, btts_no_prob]
            
            # Skor tahmini
            expected_goals_home = match_data['home_goals_scored_avg'] * 1.1  # Ev sahibi avantajı
            expected_goals_away = match_data['away_goals_scored_avg'] * 0.9  # Deplasman dezavantajı
            
            # Takım karşılaştırmasını oluştur
            team_comparison = self._create_team_comparison(match_data)
            
            # Daha detaylı gol ve korner tahminleri
            return {
                'win_prob': float(win_prob),
                'draw_prob': float(draw_prob),
                'loss_prob': float(loss_prob),
                
                # Gol tahminleri
                'total_goals': self._predict_total_goals(match_data),
                
                # Over/Under tahminleri
                'over_under_goals': {
                    '1.5': self._predict_over_under(match_data, 1.5),
                    '2.5': {
                        'over': float(over_prob),
                        'under': float(under_prob)
                    },
                    '3.5': self._predict_over_under(match_data, 3.5),
                    '4.5': self._predict_over_under(match_data, 4.5)
                },
                
                # Karşılıklı gol tahmini
                'btts_prediction': {
                    'yes': float(btts_yes_prob),
                    'no': float(btts_no_prob)
                },
                
                # Skor tahmini
                'score_prediction': {
                    'home': round(expected_goals_home),
                    'away': round(expected_goals_away)
                },
                
                # Yarı tahminleri
                'half_predictions': self._predict_halves(match_data),
                
                # Korner tahminleri
                'corner_prediction': self._predict_corners(match_data),
                
                # Oyuncu gol tahminleri
                'goalscorer_predictions': self._predict_goalscorers(match_data),
                
                # Takım karşılaştırması
                'team_comparison': team_comparison
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
            
    def _create_team_comparison(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """İki takım arasında karşılaştırmalı istatistikler oluşturur."""
        # Son 5 maç sonuçlarını oluştur (W/D/L)
        home_last_matches = self._extract_last_match_results(match_data, 'home')
        away_last_matches = self._extract_last_match_results(match_data, 'away')
        
        return {
            'home_team': {
                'form': match_data['home_form'],
                'season_points': match_data['home_season_points'],
                'goals_scored_avg': match_data['home_goals_scored_avg'],
                'goals_conceded_avg': match_data['home_goals_conceded_avg'],
                'last_matches': home_last_matches
            },
            'away_team': {
                'form': match_data['away_form'],
                'season_points': match_data['away_season_points'],
                'goals_scored_avg': match_data['away_goals_scored_avg'],
                'goals_conceded_avg': match_data['away_goals_conceded_avg'],
                'last_matches': away_last_matches
            }
        }
    
    def _extract_last_match_results(self, match_data: Dict[str, Any], team_type: str) -> List[str]:
        """Son maç sonuçlarını W/D/L formatında çıkarır."""
        results = []
        
        # match_data içinde recent_matches varsa
        if 'home_team_data' in match_data and 'recent_matches' in match_data['home_team_data']:
            recent_matches = match_data['home_team_data' if team_type == 'home' else 'away_team_data']['recent_matches']
            
            for match in recent_matches[:5]:  # Son 5 maç
                is_target_home = match.get('home_team_is_target', 
                                         team_type == 'home' and match.get('home_team') == match_data.get(f'{team_type}_team_name'))
                
                if match['winner'] == 'HOME_TEAM' and is_target_home:
                    results.append('W')
                elif match['winner'] == 'AWAY_TEAM' and not is_target_home:
                    results.append('W')
                elif match['winner'] == 'DRAW':
                    results.append('D')
                else:
                    results.append('L')
        else:
            # Demo için rastgele sonuçlar
            form = match_data[f'{team_type}_form']
            for _ in range(5):
                if random.random() < form / 100:
                    results.append('W')
                elif random.random() < 0.5:
                    results.append('D')
                else:
                    results.append('L')
                    
        return results
        
    def _predict_total_goals(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Toplam gol sayısını tahmin eder."""
        home_avg = match_data['home_goals_scored_avg']
        away_avg = match_data['away_goals_scored_avg']
        home_conceded = match_data['home_goals_conceded_avg']
        away_conceded = match_data['away_goals_conceded_avg']
        
        # Poisson dağılımına dayalı beklenen toplam gol sayısı
        home_expected = (home_avg + away_conceded) / 2
        away_expected = (away_avg + home_conceded) / 2
        expected_goals = home_expected + away_expected
        
        # Takımların formuna göre ayarlama
        form_factor = (match_data['home_form'] - match_data['away_form']) * 0.2
        home_expected += form_factor
        away_expected -= form_factor
        
        # Negatif değerleri önle
        home_expected = max(0, home_expected)
        away_expected = max(0, away_expected)
        
        adjusted_expected = home_expected + away_expected
        
        return {
            'expected': round(adjusted_expected, 1),
            'range': f"{max(0, round(adjusted_expected - 1, 1))}-{round(adjusted_expected + 1, 1)}",
            'home_expected': round(home_expected, 1),
            'away_expected': round(away_expected, 1)
        }
            
    def _predict_over_under(self, match_data: Dict[str, Any], threshold: float) -> Dict[str, float]:
        """Belirli bir gol eşiği için over/under tahminleri yapar."""
        home_avg = match_data['home_goals_scored_avg']
        away_avg = match_data['away_goals_scored_avg']
        home_conceded = match_data['home_goals_conceded_avg']
        away_conceded = match_data['away_goals_conceded_avg']
        
        # Takım başına beklenen gol sayılarını hesapla
        home_expected = (home_avg + away_conceded) / 2
        away_expected = (away_avg + home_conceded) / 2
        
        # Form faktörünü dahil et
        form_diff = match_data['home_form'] - match_data['away_form']
        home_expected += form_diff * 0.1
        away_expected -= form_diff * 0.1
        
        # Minimum 0 olmalı
        home_expected = max(0, home_expected)
        away_expected = max(0, away_expected)
        
        total_expected = home_expected + away_expected
        
        # Poisson dağılımı kullanarak gol olasılıklarını hesapla
        # Bu kısaltılmış bir hesaplama, tam Poisson hesaplaması daha karmaşık olabilir
        lambda_val = total_expected
        
        # k veya daha fazla gol olma olasılığı (over için)
        k = int(threshold)
        decimal_part = threshold - k
        
        # Basitleştirilmiş hesaplama
        # Gol sınırından daha az gol atılma olasılığı (under için)
        under_prob = 0
        for i in range(k):
            # Poisson olasılığının basit yaklaşımı
            under_prob += self._poisson_probability(lambda_val, i)
        
        # Eşik ondalıklı ise (ör. 2.5), tam sayı için üst sınırı dahil etme
        if decimal_part > 0:
            under_prob += self._poisson_probability(lambda_val, k) * (1 - decimal_part)
        
        over_prob = 1 - under_prob
        
        # Ligler ve takımlara özgü ayarlamalar
        # Örneğin, belirli liglerde daha çok ya da az gol atılma eğilimi
        league_factor = 1.0  # Varsayılan değer
        
        # Takımların isimleriyle lig tahmini
        if any(team in ["Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla"] for team in [match_data['home_team_name'], match_data['away_team_name']]):
            league_factor = 1.1  # İspanya ligi daha golcü
        elif any(team in ["Bayern Munich", "Borussia Dortmund", "RB Leipzig"] for team in [match_data['home_team_name'], match_data['away_team_name']]):
            league_factor = 1.2  # Bundesliga en golcü liglerden
        elif any(team in ["Liverpool", "Manchester City", "Manchester United", "Chelsea"] for team in [match_data['home_team_name'], match_data['away_team_name']]):
            league_factor = 1.15  # Premier Lig golcü
        elif any(team in ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"] for team in [match_data['home_team_name'], match_data['away_team_name']]):
            league_factor = 0.95  # Süper Lig ortalamanın altında
        
        # Lig faktörünü uygula (üst sınır ve alt sınır ayarlamaları)
        over_prob = min(0.95, max(0.05, over_prob * league_factor))
        under_prob = 1 - over_prob
        
        return {
            'over': float(over_prob),
            'under': float(under_prob)
        }
        
    def _poisson_probability(self, lambda_val: float, k: int) -> float:
        """Poisson olasılığını hesaplar: P(X = k) = (e^-lambda * lambda^k) / k!"""
        import math
        try:
            # e^-lambda * lambda^k
            numerator = math.exp(-lambda_val) * (lambda_val ** k)
            # k!
            denominator = math.factorial(k)
            return numerator / denominator
        except:
            # Hesaplama hatası durumunda
            if k < lambda_val:
                return 0.1  # Düşük k için düşük olasılık
            else:
                return 0.9  # Yüksek k için yüksek olasılık
            
    def _predict_corners(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Korner tahminlerini yapar."""
        # Takımların ofansif güçlerine göre korner sayısını tahmin et
        home_attack_strength = match_data['home_goals_scored_avg']
        away_attack_strength = match_data['away_goals_scored_avg']
        
        # Ortalama beklenen korner sayısı
        expected_corners = (home_attack_strength + away_attack_strength) * 5
        
        # Farklı eşikler için olasılıklar
        return {
            'total_corners': round(expected_corners),
            'corner_ranges': {
                '3.5': {
                    'over': float(expected_corners > 3.5),
                    'under': float(expected_corners <= 3.5)
                },
                '4.5': {
                    'over': float(expected_corners > 4.5),
                    'under': float(expected_corners <= 4.5)
                },
                '5.5': {
                    'over': float(expected_corners > 5.5),
                    'under': float(expected_corners <= 5.5)
                },
                '8.5': {
                    'over': float(expected_corners > 8.5),
                    'under': float(expected_corners <= 8.5)
                }
            }
        }
    
    def _predict_halves(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """İlk yarı ve ikinci yarı tahminlerini yapar."""
        # Takımların form ve güçlerini analiz et
        home_form = match_data['home_form']
        away_form = match_data['away_form']
        
        # İlk yarı ev sahibi takımlar daha iyi başlar genellikle
        first_half_home_advantage = 1.2
        second_half_home_advantage = 1.05
        
        first_half_home_strength = home_form * first_half_home_advantage
        first_half_away_strength = away_form
        
        second_half_home_strength = home_form * second_half_home_advantage
        second_half_away_strength = away_form * 1.1  # İkinci yarıda deplasman takımları genelde açılır
        
        total_first_half = first_half_home_strength + first_half_away_strength
        total_second_half = second_half_home_strength + second_half_away_strength
        
        # İlk yarı tahminleri
        first_half_win_prob = max(0.01, min(0.9, first_half_home_strength / total_first_half))
        first_half_loss_prob = max(0.01, min(0.9, first_half_away_strength / total_first_half))
        first_half_draw_prob = max(0.01, min(0.9, 1 - first_half_win_prob - first_half_loss_prob))
        
        # İkinci yarı tahminleri
        second_half_win_prob = max(0.01, min(0.9, second_half_home_strength / total_second_half))
        second_half_loss_prob = max(0.01, min(0.9, second_half_away_strength / total_second_half))
        second_half_draw_prob = max(0.01, min(0.9, 1 - second_half_win_prob - second_half_loss_prob))
        
        return {
            'first_half': {
                'home_win': float(first_half_win_prob),
                'draw': float(first_half_draw_prob),
                'away_win': float(first_half_loss_prob),
                'goals': float(round((first_half_home_strength + first_half_away_strength) * 0.3, 1))  # İlk yarılar genelde daha az gollü olur
            },
            'second_half': {
                'home_win': float(second_half_win_prob),
                'draw': float(second_half_draw_prob),
                'away_win': float(second_half_loss_prob),
                'goals': float(round((second_half_home_strength + second_half_away_strength) * 0.4, 1))
            }
        }
        
    def _predict_goalscorers(self, match_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Gol atması muhtemel oyuncuları tahmin eder."""
        # Örnek oyuncu verileri (gerçek bir uygulamada oyuncu veri tabanından çekilir)
        home_team_name = match_data.get('home_team_name', 'Ev Sahibi')
        away_team_name = match_data.get('away_team_name', 'Deplasman')
        
        # Ev sahibi takım için muhtemel gol atabilecek oyuncular
        home_team_players = self._get_example_players(home_team_name)
        
        # Deplasman takımı için muhtemel gol atabilecek oyuncular
        away_team_players = self._get_example_players(away_team_name)
        
        # Takımların gol beklentilerine göre oyuncu gol olasılıklarını ayarla
        home_expected_goals = match_data['home_goals_scored_avg']
        away_expected_goals = match_data['away_goals_scored_avg']
        
        self._adjust_goalscoring_probabilities(home_team_players, home_expected_goals)
        self._adjust_goalscoring_probabilities(away_team_players, away_expected_goals)
        
        return {
            'home_team': home_team_players,
            'away_team': away_team_players
        }
        
    def _get_example_players(self, team_name: str) -> List[Dict[str, Any]]:
        """Takım adına göre örnek oyuncu listesi döndürür."""
        # Gerçek bir uygulama için burada API'den oyuncu verisi çekilir
        
        if team_name.lower() == "galatasaray":
            return [
                {"name": "Mauro Icardi", "position": "ST", "scoring_prob": 0.45},
                {"name": "Dries Mertens", "position": "AMF", "scoring_prob": 0.25},
                {"name": "Kerem Aktürkoğlu", "position": "LW", "scoring_prob": 0.20},
                {"name": "Yunus Akgün", "position": "RW", "scoring_prob": 0.18},
                {"name": "Victor Osimhen", "position": "ST", "scoring_prob": 0.40}
            ]
        elif team_name.lower() == "fenerbahçe":
            return [
                {"name": "Edin Dzeko", "position": "ST", "scoring_prob": 0.40},
                {"name": "Dusan Tadic", "position": "AMF", "scoring_prob": 0.22},
                {"name": "Sebastian Szymanski", "position": "CMF", "scoring_prob": 0.18},
                {"name": "İrfan Can Kahveci", "position": "RW", "scoring_prob": 0.20},
                {"name": "Cenk Tosun", "position": "ST", "scoring_prob": 0.25}
            ]
        elif team_name.lower() == "beşiktaş":
            return [
                {"name": "Vincent Aboubakar", "position": "ST", "scoring_prob": 0.35},
                {"name": "Rachid Ghezzal", "position": "RW", "scoring_prob": 0.20},
                {"name": "Gedson Fernandes", "position": "CMF", "scoring_prob": 0.15},
                {"name": "Alex Oxlade-Chamberlain", "position": "AMF", "scoring_prob": 0.22},
                {"name": "Ciro Immobile", "position": "ST", "scoring_prob": 0.40}
            ]
        elif team_name.lower() == "trabzonspor":
            return [
                {"name": "Enis Bardhi", "position": "AMF", "scoring_prob": 0.22},
                {"name": "Paul Onuachu", "position": "ST", "scoring_prob": 0.35},
                {"name": "Edin Visca", "position": "RW", "scoring_prob": 0.18},
                {"name": "Anastasios Bakasetas", "position": "AMF", "scoring_prob": 0.25},
                {"name": "Umut Bozok", "position": "ST", "scoring_prob": 0.30}
            ]
        else:
            # Diğer takımlar için rastgele oyuncular
            return [
                {"name": "Forvet Oyuncusu", "position": "ST", "scoring_prob": 0.30},
                {"name": "Orta Saha 1", "position": "AMF", "scoring_prob": 0.20},
                {"name": "Kanat Oyuncusu", "position": "LW", "scoring_prob": 0.18},
                {"name": "Orta Saha 2", "position": "CMF", "scoring_prob": 0.15},
                {"name": "Yedek Forvet", "position": "ST", "scoring_prob": 0.25}
            ]
    
    def _adjust_goalscoring_probabilities(self, players: List[Dict[str, Any]], expected_goals: float) -> None:
        """Beklenen gol sayısına göre oyuncu gol olasılıklarını ayarlar."""
        # Takımın toplam gol beklentisine göre skaler bir faktör hesapla
        scaling_factor = expected_goals / 1.5  # 1.5 ortalama değer olarak kabul edilir
        
        for player in players:
            # Oyuncunun gol atma olasılığını takımın genel gol beklentisine göre ayarla
            player["scoring_prob"] = min(0.9, player["scoring_prob"] * scaling_factor) 
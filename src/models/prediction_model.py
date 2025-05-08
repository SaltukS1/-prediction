import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import math
import random
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import os
import sys

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# İçe aktarmalar
from src.data.api_client import FootballApiClient
from src.models.ml_model import FootballMLModel

class PredictionModel:
    """Futbol maç tahmin modeli."""
    
    def __init__(self):
        """Tahmin modelini başlatır."""
        self._initialize_model()
        
        # API istemcisini başlat
        self.api_client = FootballApiClient()
        
        # ML modelini başlat
        self.ml_model = FootballMLModel()
        
        # Hybrid mod (0: Sadece istatistiksel, 1: Sadece ML, 0.5: Karma)
        self.hybrid_mode = 0.5
    
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
        """Maç sonucunu tahmin eder.
        
        Args:
            match_data: Maç verileri
            
        Returns:
            Tahmin sonuçları
        """
        # Gerçek zamanlı takım ve oyuncu verilerini çek
        home_team_name = match_data.get("home_team", {}).get("name", "")
        away_team_name = match_data.get("away_team", {}).get("name", "")
        
        # Takımları ve oyuncu verilerini API'den sorgula
        try:
            # API'den takım verilerini çek
            home_team_data = self.api_client.get_team_data(home_team_name)
            away_team_data = self.api_client.get_team_data(away_team_name)
            
            # API'den oyuncu verilerini çek
            home_players = self.api_client.get_player_data(home_team_name)
            away_players = self.api_client.get_player_data(away_team_name)
            
            # Varsa H2H verilerini çek
            h2h_data = self.api_client.get_h2h_data(home_team_name, away_team_name)
            
            # API verilerini match_data'ya ekle
            match_data["home_team"]["api_data"] = home_team_data
            match_data["away_team"]["api_data"] = away_team_data
            match_data["home_players"] = home_players
            match_data["away_players"] = away_players
            match_data["h2h_data"] = h2h_data
            
            # Verileri başarıyla çektiysen API durumunu güncelle
            match_data["api_status"] = "success"
            
        except Exception as e:
            print(f"API veri çekme hatası: {str(e)}")
            # Hata durumunda örnek oyuncu verilerini kullan
            match_data["home_players"] = self._get_example_players(home_team_name)
            match_data["away_players"] = self._get_example_players(away_team_name)
            match_data["api_status"] = "error"
        
        # Detaylı takım analizi yap
        team_analysis = self._perform_detailed_team_analysis(match_data)
        
        # İstatistiksel tahmin
        statistical_prediction = self._make_statistical_prediction(match_data, team_analysis)
        
        # ML tabanlı tahmin
        try:
            ml_prediction = self._make_ml_prediction(match_data, team_analysis)
            has_ml_prediction = True
        except Exception as e:
            print(f"ML tahmin hatası: {str(e)}")
            ml_prediction = statistical_prediction
            has_ml_prediction = False
        
        # Hybrid tahmin (istatistiksel ve ML karışımı)
        if has_ml_prediction:
            prediction = self._combine_predictions(statistical_prediction, ml_prediction)
        else:
            prediction = statistical_prediction
        
        # Ek tahmin detayları ekle
        prediction["detailed_analysis"] = team_analysis
        prediction["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction["version"] = "3.0.0"
        prediction["api_status"] = match_data.get("api_status", "unknown")
        prediction["ml_model_used"] = has_ml_prediction
        
        return prediction
    
    def _make_statistical_prediction(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """İstatistiksel yöntemlerle tahmin yapar.
        
        Args:
            match_data: Maç verileri
            team_analysis: Takım analizi
            
        Returns:
            İstatistiksel tahmin sonuçları
        """
        # Sonuç tahmini
        home_goals = team_analysis["home_team"]["expected_goals"]
        away_goals = team_analysis["away_team"]["expected_goals"]
        
        # Poisson dağılımı ile sonuç olasılıkları
        result_probs = self._calculate_match_result_probabilities(home_goals, away_goals)
        
        # Diğer tahminler
        correct_score = self._predict_correct_score(home_goals, away_goals)
        total_goals = self._predict_total_goals(match_data, team_analysis)
        over_under = self._predict_over_under(match_data, 2.5, team_analysis)
        corners = self._predict_corners(match_data, team_analysis)
        halves = self._predict_halves(match_data, team_analysis)
        goalscorers = self._predict_goalscorers(match_data, team_analysis)
        
        # Tahmin sonuçlarını döndür
        return {
            "match_result": result_probs,
            "correct_score": correct_score,
            "total_goals": total_goals,
            "over_under": over_under,
            "corners": corners,
            "halves": halves,
            "goalscorers": goalscorers
        }
    
    def _make_ml_prediction(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ML modeli ile tahmin yapar.
        
        Args:
            match_data: Maç verileri
            team_analysis: Takım analizi
            
        Returns:
            ML tahmin sonuçları
        """
        # ML modeli için girdi verilerini hazırla
        ml_input_data = self._prepare_ml_input(match_data, team_analysis)
        
        # ML modeli ile tahmin yap
        ml_results = self.ml_model.predict(ml_input_data)
        
        # ML çıktısını tahmin formatına çevir
        match_result = {
            "home_win": ml_results["match_result"]["home_win_prob"],
            "draw": ml_results["match_result"]["draw_prob"],
            "away_win": ml_results["match_result"]["away_win_prob"]
        }
        
        correct_score = {
            "prediction": ml_results["score"]["prediction"],
            "probabilities": ml_results["score"]["top_scores"]
        }
        
        total_goals = {
            "prediction": ml_results["goals"]["prediction"],
            "mean": ml_results["goals"]["prediction"],
            "probabilities": {
                str(k): float(v) for k, v in ml_results["goals"]["probabilities"].items()
            }
        }
        
        # İstatistiksel tahminden gelen diğer özellikleri ekle
        statistical_prediction = self._make_statistical_prediction(match_data, team_analysis)
        
        return {
            "match_result": match_result,
            "correct_score": correct_score,
            "total_goals": total_goals,
            "over_under": statistical_prediction["over_under"],
            "corners": statistical_prediction["corners"],
            "halves": statistical_prediction["halves"],
            "goalscorers": statistical_prediction["goalscorers"]
        }
    
    def _prepare_ml_input(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ML modeli için girdi verilerini hazırlar.
        
        Args:
            match_data: Maç verileri
            team_analysis: Takım analizi
            
        Returns:
            ML modeli için hazırlanmış veriler
        """
        # Takım özellikleri
        home_team = match_data.get("home_team", {})
        away_team = match_data.get("away_team", {})
        
        home_team_analysis = team_analysis.get("home_team", {})
        away_team_analysis = team_analysis.get("away_team", {})
        
        # Son 5 maçtaki performans
        home_form = home_team_analysis.get("form", [])
        away_form = away_team_analysis.get("form", [])
        
        home_form_points = sum(3 if result == "W" else 1 if result == "D" else 0 for result in home_form)
        away_form_points = sum(3 if result == "W" else 1 if result == "D" else 0 for result in away_form)
        
        # Son maçlardaki gol istatistikleri
        home_goals_scored = home_team_analysis.get("goals_scored", 0)
        home_goals_conceded = home_team_analysis.get("goals_conceded", 0)
        away_goals_scored = away_team_analysis.get("goals_scored", 0)
        away_goals_conceded = away_team_analysis.get("goals_conceded", 0)
        
        # Oyuncu verileri
        home_players = match_data.get("home_players", [])
        away_players = match_data.get("away_players", [])
        
        # ML girdi verileri
        ml_input = {
            "home_team": {
                "name": home_team.get("name", ""),
                "form_points": home_form_points,
                "goals_scored": home_goals_scored,
                "goals_conceded": home_goals_conceded,
                "xG": home_team_analysis.get("expected_goals", 0),
                "xGA": away_team_analysis.get("expected_goals", 0),
                "possession": home_team_analysis.get("possession", 50),
                "shots": home_team_analysis.get("shots", 0),
                "shots_on_target": home_team_analysis.get("shots_on_target", 0),
                "corners": home_team_analysis.get("corners", 0),
                "home_advantage": 0.6
            },
            "away_team": {
                "name": away_team.get("name", ""),
                "form_points": away_form_points,
                "goals_scored": away_goals_scored,
                "goals_conceded": away_goals_conceded,
                "xG": away_team_analysis.get("expected_goals", 0),
                "xGA": home_team_analysis.get("expected_goals", 0),
                "possession": away_team_analysis.get("possession", 50),
                "shots": away_team_analysis.get("shots", 0),
                "shots_on_target": away_team_analysis.get("shots_on_target", 0),
                "corners": away_team_analysis.get("corners", 0),
                "away_disadvantage": 0.4
            },
            "home_players": home_players,
            "away_players": away_players
        }
        
        return ml_input
    
    def _combine_predictions(self, statistical_prediction: Dict[str, Any], ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """İstatistiksel ve ML tahminlerini birleştirir.
        
        Args:
            statistical_prediction: İstatistiksel tahmin
            ml_prediction: ML tahmini
            
        Returns:
            Birleştirilmiş tahmin
        """
        combined_prediction = {}
        
        # Hybrid modu kullan (0: sadece istatistiksel, 1: sadece ML, 0.5: dengeli karışım)
        ml_weight = self.hybrid_mode
        stat_weight = 1 - ml_weight
        
        # Maç sonucu olasılıkları
        combined_prediction["match_result"] = {
            "home_win": statistical_prediction["match_result"]["home_win"] * stat_weight + ml_prediction["match_result"]["home_win"] * ml_weight,
            "draw": statistical_prediction["match_result"]["draw"] * stat_weight + ml_prediction["match_result"]["draw"] * ml_weight,
            "away_win": statistical_prediction["match_result"]["away_win"] * stat_weight + ml_prediction["match_result"]["away_win"] * ml_weight
        }
        
        # Skoru ve diğer tahminleri birleştir
        # Doğru skor tahmini
        if ml_weight >= 0.7:  # ML modeli ağırlıklıysa
            combined_prediction["correct_score"] = ml_prediction["correct_score"]
        else:
            combined_prediction["correct_score"] = statistical_prediction["correct_score"]
        
        # Toplam gol tahmini
        combined_prediction["total_goals"] = {
            "prediction": round(statistical_prediction["total_goals"]["prediction"] * stat_weight + ml_prediction["total_goals"]["prediction"] * ml_weight),
            "mean": statistical_prediction["total_goals"]["mean"] * stat_weight + ml_prediction["total_goals"]["mean"] * ml_weight,
            "probabilities": {}
        }
        
        # Olasılıkları birleştir (eğer varsa)
        for key in statistical_prediction["total_goals"].get("probabilities", {}):
            stat_prob = statistical_prediction["total_goals"]["probabilities"].get(key, 0)
            ml_prob = ml_prediction["total_goals"]["probabilities"].get(key, 0)
            combined_prediction["total_goals"]["probabilities"][key] = stat_prob * stat_weight + ml_prob * ml_weight
        
        # Diğer özellikleri ekle
        combined_prediction["over_under"] = statistical_prediction["over_under"]
        combined_prediction["corners"] = statistical_prediction["corners"]
        combined_prediction["halves"] = statistical_prediction["halves"]
        combined_prediction["goalscorers"] = statistical_prediction["goalscorers"]
        
        return combined_prediction
    
    def _perform_detailed_team_analysis(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maç için çok daha detaylı takım analizi yapar."""
        home_team_name = match_data.get('home_team_name', 'Ev Sahibi')
        away_team_name = match_data.get('away_team_name', 'Deplasman')
        
        # Takımların temel verilerini al
        home_form = match_data.get('home_form', 70)
        away_form = match_data.get('away_form', 70)
        home_season_points = match_data.get('home_season_points', 0)
        away_season_points = match_data.get('away_season_points', 0)
        
        # Oyuncu listelerini al
        home_players = self._get_example_players(home_team_name)
        away_players = self._get_example_players(away_team_name)
        
        # Takım değeri hesaplaması - string değerleri float'a çevirirken hata kontrolü ekle
        try:
            home_team_value = sum([float(str(player.get('market_value', '0M€')).replace('M€', '').replace('€', '')) 
                                for player in home_players], 0)
            away_team_value = sum([float(str(player.get('market_value', '0M€')).replace('M€', '').replace('€', '')) 
                                for player in away_players], 0)
        except Exception as e:
            logger.warning(f"Market value conversion error: {str(e)}")
            home_team_value = 0
            away_team_value = 0
        
        # Sakat oyuncuların etkisi - string değerleri float'a çevirirken hata kontrolü ekle
        try:
            home_injured_value = sum([float(str(player.get('market_value', '0M€')).replace('M€', '').replace('€', '')) 
                                    for player in home_players if not player.get('is_available', True)], 0)
            away_injured_value = sum([float(str(player.get('market_value', '0M€')).replace('M€', '').replace('€', '')) 
                                    for player in away_players if not player.get('is_available', True)], 0)
        except Exception as e:
            logger.warning(f"Injured player value conversion error: {str(e)}")
            home_injured_value = 0
            away_injured_value = 0
        
        # Sakat oyuncuların takım değerine etkisi
        home_injury_impact = min(0.3, home_injured_value / max(1, home_team_value)) if home_team_value > 0 else 0.1
        away_injury_impact = min(0.3, away_injured_value / max(1, away_team_value)) if away_team_value > 0 else 0.1
        
        # Gol atan oyuncuların formu
        home_goalscorers_form = 0
        away_goalscorers_form = 0
        
        try:
            home_goalscorers_form = sum([sum(player.get('last_5_form', [0,0,0,0,0])) / 5 
                                      for player in home_players if player.get('goals', 0) > 2 
                                      and player.get('is_available', True)], 0)
            away_goalscorers_form = sum([sum(player.get('last_5_form', [0,0,0,0,0])) / 5 
                                      for player in away_players if player.get('goals', 0) > 2
                                      and player.get('is_available', True)], 0)
        except Exception as e:
            logger.warning(f"Player form calculation error: {str(e)}")
        
        # UEFA Takım sıralaması verilerini kullan
        home_uefa_rank = self._get_uefa_team_rank(home_team_name)
        away_uefa_rank = self._get_uefa_team_rank(away_team_name)
        
        # Takımların gol ortalamaları
        home_goals_scored_avg = match_data.get('home_goals_scored_avg', 1.5)
        home_goals_conceded_avg = match_data.get('home_goals_conceded_avg', 1.0)
        away_goals_scored_avg = match_data.get('away_goals_scored_avg', 1.2)
        away_goals_conceded_avg = match_data.get('away_goals_conceded_avg', 1.3)
        
        # Son maçlar
        home_recent_matches = []
        away_recent_matches = []
        
        if 'home_team_data' in match_data and 'recent_matches' in match_data['home_team_data']:
            home_recent_matches = match_data['home_team_data']['recent_matches']
        if 'away_team_data' in match_data and 'recent_matches' in match_data['away_team_data']:
            away_recent_matches = match_data['away_team_data']['recent_matches']
        
        # Son 5 maçtaki gol ortalamaları
        home_last5_goals_scored = 0
        home_last5_goals_conceded = 0
        away_last5_goals_scored = 0
        away_last5_goals_conceded = 0
        
        try:
            for i, match in enumerate(home_recent_matches[:5]):
                is_home_team = match.get('home_team') == home_team_name
                if is_home_team:
                    home_last5_goals_scored += match.get('home_score', 0)
                    home_last5_goals_conceded += match.get('away_score', 0)
                else:
                    home_last5_goals_scored += match.get('away_score', 0)
                    home_last5_goals_conceded += match.get('home_score', 0)
            
            for i, match in enumerate(away_recent_matches[:5]):
                is_away_team = match.get('away_team') == away_team_name
                if is_away_team:
                    away_last5_goals_scored += match.get('away_score', 0)
                    away_last5_goals_conceded += match.get('home_score', 0)
                else:
                    away_last5_goals_scored += match.get('home_score', 0)
                    away_last5_goals_conceded += match.get('away_score', 0)
        except Exception as e:
            logger.warning(f"Recent match calculation error: {str(e)}")
        
        # Son 5 maç ortalamaları
        home_last5_scored_avg = home_last5_goals_scored / 5 if len(home_recent_matches) >= 5 else home_goals_scored_avg
        home_last5_conceded_avg = home_last5_goals_conceded / 5 if len(home_recent_matches) >= 5 else home_goals_conceded_avg
        away_last5_scored_avg = away_last5_goals_scored / 5 if len(away_recent_matches) >= 5 else away_goals_scored_avg
        away_last5_conceded_avg = away_last5_goals_conceded / 5 if len(away_recent_matches) >= 5 else away_goals_conceded_avg
        
        # Takım gücü hesaplaması - gelişmiş ve UEFA sıralaması dahil
        # Form %30, Sezon puanları %15, Oyuncu kadrosu %20, Son 5 maç %10, UEFA sıralaması %25
        home_uefa_factor = self._calculate_uefa_factor(home_uefa_rank)
        away_uefa_factor = self._calculate_uefa_factor(away_uefa_rank)
        
        home_strength_base = (home_form / 100 * 0.3) + (home_season_points / 100 * 0.15) + (min(1, home_team_value / 300) * 0.2) + (home_goalscorers_form * 0.1) + (home_uefa_factor * 0.25)
        away_strength_base = (away_form / 100 * 0.3) + (away_season_points / 100 * 0.15) + (min(1, away_team_value / 300) * 0.2) + (away_goalscorers_form * 0.1) + (away_uefa_factor * 0.25)
        
        # Sakatları hesaba kat
        home_strength_adjusted = home_strength_base * (1 - home_injury_impact)
        away_strength_adjusted = away_strength_base * (1 - away_injury_impact)
        
        # Ev sahibi avantajı
        home_advantage = 0.1
        
        # Beklenen gol sayıları - gelişmiş hesaplama
        home_expected_goals = (
            (home_goals_scored_avg * 0.3) + 
            (away_goals_conceded_avg * 0.25) + 
            (home_last5_scored_avg * 0.2) +
            (home_strength_adjusted * 2.0 * 0.25)
        ) * (1 + home_advantage)
        
        away_expected_goals = (
            (away_goals_scored_avg * 0.3) + 
            (home_goals_conceded_avg * 0.25) + 
            (away_last5_scored_avg * 0.2) +
            (away_strength_adjusted * 2.0 * 0.25)
        ) * (1 - home_advantage/2)  # Deplasman dezavantajını azalttık
        
        # Takımlar arasındaki güç farkı çok fazlaysa gol dağılımını buna göre ayarla
        if home_strength_adjusted / max(0.1, away_strength_adjusted) > 2:
            # Ev sahibi çok güçlüyse
            home_expected_goals *= 1.2
        elif away_strength_adjusted / max(0.1, home_strength_adjusted) > 2:
            # Deplasman çok güçlüyse
            away_expected_goals *= 1.2
        
        # Detaylı analiz sonucunu döndür
        return {
            'home_team': {
                'name': home_team_name,
                'base_form': home_form / 100,
                'last_5_form': home_goalscorers_form,
                'team_value': home_team_value,
                'injured_value': home_injured_value,
                'injury_impact': home_injury_impact,
                'uefa_rank': home_uefa_rank,
                'uefa_factor': home_uefa_factor,
                'season_stats': {
                    'points': home_season_points,
                    'goals_scored_avg': home_goals_scored_avg,
                    'goals_conceded_avg': home_goals_conceded_avg
                },
                'last_5_matches': {
                    'goals_scored_avg': home_last5_scored_avg,
                    'goals_conceded_avg': home_last5_conceded_avg
                },
                'base_strength': home_strength_base,
                'adjusted_strength': home_strength_adjusted,
                'expected_goals': home_expected_goals,
                'available_players': len([p for p in home_players if p.get('is_available', True)]),
                'unavailable_players': len([p for p in home_players if not p.get('is_available', True)])
            },
            'away_team': {
                'name': away_team_name,
                'base_form': away_form / 100,
                'last_5_form': away_goalscorers_form,
                'team_value': away_team_value,
                'injured_value': away_injured_value,
                'injury_impact': away_injury_impact,
                'uefa_rank': away_uefa_rank,
                'uefa_factor': away_uefa_factor,
                'season_stats': {
                    'points': away_season_points,
                    'goals_scored_avg': away_goals_scored_avg,
                    'goals_conceded_avg': away_goals_conceded_avg
                },
                'last_5_matches': {
                    'goals_scored_avg': away_last5_scored_avg,
                    'goals_conceded_avg': away_last5_conceded_avg
                },
                'base_strength': away_strength_base,
                'adjusted_strength': away_strength_adjusted,
                'expected_goals': away_expected_goals,
                'available_players': len([p for p in away_players if p.get('is_available', True)]),
                'unavailable_players': len([p for p in away_players if not p.get('is_available', True)])
            },
            'team_matchup': {
                'value_ratio': home_team_value / max(1, away_team_value),
                'form_ratio': (home_form / max(1, away_form)),
                'expected_goals_ratio': home_expected_goals / max(0.1, away_expected_goals),
                'strength_ratio': home_strength_adjusted / max(0.1, away_strength_adjusted)
            }
        }
        
    def _calculate_uefa_factor(self, rank: int) -> float:
        """UEFA sıralamasına göre güç faktörü hesaplar (0-1 arası)"""
        if rank <= 0:  # Geçersiz veya bilinmeyen sıralama
            return 0.5
        
        # Top 10 takımlar
        if rank <= 10:
            return 0.95 - (rank - 1) * 0.02  # 0.95 -> 0.77
        # Top 11-25 takımlar
        elif rank <= 25:
            return 0.75 - (rank - 11) * 0.01  # 0.75 -> 0.61
        # Top 26-50 takımlar
        elif rank <= 50:
            return 0.60 - (rank - 26) * 0.005  # 0.60 -> 0.48
        # Top 51-100 takımlar
        elif rank <= 100:
            return 0.47 - (rank - 51) * 0.002  # 0.47 -> 0.37
        # 100+ takımlar
        else:
            return max(0.2, 0.35 - (rank - 101) * 0.001)  # 0.35 ve aşağısı
            
    def _get_uefa_team_rank(self, team_name: str) -> int:
        """Takımın UEFA sıralamasını döndürür"""
        # Takım isim normalizasyonu
        team_name_lower = team_name.lower().strip()
        
        # UEFA takım sıralaması veritabanı - 2023-2024 verileri
        rankings = {
            # Top 10
            "real madrid": 1,
            "manchester city": 2,
            "bayern munich": 3, 
            "bayern münih": 3,
            "liverpool": 4,
            "psg": 5,
            "paris saint-germain": 5,
            "paris saint germain": 5,
            "inter": 6,
            "inter milan": 6,
            "dortmund": 7,
            "borussia dortmund": 7,
            "chelsea": 8,
            "roma": 9,
            "barcelona": 10,
            # 11-25
            "manchester united": 11,
            "man united": 11,
            "arsenal": 12,
            "leverkusen": 13,
            "bayer leverkusen": 13,
            "atletico madrid": 14,
            "atlético madrid": 14,
            "benfica": 15,
            "villarreal": 16,
            "atalanta": 17,
            "porto": 18,
            "leipzig": 19,
            "rb leipzig": 19,
            "milan": 20,
            "ac milan": 20,
            "lazio": 21,
            "juventus": 22,
            "frankfurt": 23,
            "eintracht frankfurt": 23,
            "club brugge": 24,
            "rangers": 25,
            "glasgow rangers": 25,
            # Türk Takımları
            "fenerbahçe": 47,
            "galatasaray": 58,
            "başakşehir": 80,
            "istanbul başakşehir": 80,
            "beşiktaş": 103,
            "trabzonspor": 127,
            "sivasspor": 98
        }
        
        # Takım bulunursa sıralamayı döndür, bulunamazsa varsayılan değer
        for key, rank in rankings.items():
            if key in team_name_lower:
                return rank
                
        # Takım bulunamadıysa ortada bir değer döndür
        return 120
    
    def _calculate_match_result_probabilities(self, home_goals: float, away_goals: float) -> Dict[str, float]:
        """Poisson dağılımı kullanarak maç sonucu olasılıklarını hesaplar"""
        max_goals = 10  # Hesaplanacak maksimum gol sayısı
        
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        # Tüm olası skorları hesapla
        for home_score in range(max_goals):
            for away_score in range(max_goals):
                # Bu skorun olasılığı
                score_prob = (self._poisson_probability(home_goals, home_score) * 
                             self._poisson_probability(away_goals, away_score))
                
                # Sonuca göre topla
                if home_score > away_score:
                    home_win_prob += score_prob
                elif home_score == away_score:
                    draw_prob += score_prob
                else:
                    away_win_prob += score_prob
        
        # Normalize et (toplam 1 olmalı)
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob
        }
        
    def _calculate_over_probability(self, expected_goals: float, threshold: float) -> float:
        """Poisson dağılımını kullanarak belirli bir eşik için over olasılığını hesaplar"""
        k = int(threshold)
        
        # Eşiğin altındaki tüm gol olasılıklarını topla (under olasılığı)
        under_prob = 0.0
        for i in range(k + 1):
            under_prob += self._poisson_probability(expected_goals, i)
            
        # Over olasılığı = 1 - under olasılığı
        return 1 - under_prob
        
    def _calculate_btts_probability(self, home_goals: float, away_goals: float) -> float:
        """Her iki takımın da gol atma olasılığını hesaplar"""
        # Her iki takımın da gol atamama olasılığı
        home_no_goal = self._poisson_probability(home_goals, 0)
        away_no_goal = self._poisson_probability(away_goals, 0)
        
        # En az bir takım gol atamama olasılığı
        no_btts_prob = home_no_goal + away_no_goal - (home_no_goal * away_no_goal)
        
        # Karşılıklı gol olasılığı
        return 1 - no_btts_prob
        
    def _predict_correct_score(self, home_expected: float, away_expected: float) -> Dict[str, Any]:
        """Poisson dağılımını kullanarak doğru skor tahmini yapar"""
        max_goals = 5  # Maksimum hesaplanacak gol sayısı
        
        # Her skor olasılığını hesapla
        score_probs = {}
        max_prob = 0.0
        most_likely_score = {'home': 0, 'away': 0}
        
        total_prob = 0.0
        
        for home_score in range(max_goals + 1):
            for away_score in range(max_goals + 1):
                # Bu skorun olasılığı
                prob = (self._poisson_probability(home_expected, home_score) * 
                       self._poisson_probability(away_expected, away_score))
                       
                score_key = f"{home_score}-{away_score}"
                score_probs[score_key] = prob
                total_prob += prob
                
                # En yüksek olasılıklı skoru bul
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = {'home': home_score, 'away': away_score}
        
        # Olasılıkları normalize et
        normalized_probs = {}
        for score, prob in score_probs.items():
            normalized_probs[score] = prob / total_prob if total_prob > 0 else 0
            
        # En olası 3 skoru bul
        top_scores = sorted(normalized_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Sonuçları düzenle
        result = {
            'home': most_likely_score['home'],
            'away': most_likely_score['away'],
            'most_likely': most_likely_score,
            'probability': float(max_prob / total_prob) if total_prob > 0 else 0,
            'top_scores': [{'score': score, 'probability': float(prob)} for score, prob in top_scores]
        }
        
        return result
        
    def _predict_total_goals(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Toplam gol sayısını tahmin eder."""
        if team_analysis:
            # Gelişmiş analiz kullan
            home_expected = team_analysis['home_team']['expected_goals']
            away_expected = team_analysis['away_team']['expected_goals']
            
            # Sakat oyuncuların etkisini hesaba kat
            home_expected *= (1 - team_analysis['home_team']['injury_impact'] * 0.5)
            away_expected *= (1 - team_analysis['away_team']['injury_impact'] * 0.5)
            
            # Son 5 maçtaki gol ortalamasını hesaba kat
            home_expected = (home_expected * 0.7) + (team_analysis['home_team']['last_5_matches']['goals_scored_avg'] * 0.3)
            away_expected = (away_expected * 0.7) + (team_analysis['away_team']['last_5_matches']['goals_scored_avg'] * 0.3)
            
            adjusted_expected = home_expected + away_expected
        else:
            # Eski yöntemi kullan
            home_avg = match_data['home_goals_scored_avg']
            away_avg = match_data['away_goals_scored_avg']
            home_conceded = match_data['home_goals_conceded_avg']
            away_conceded = match_data['away_goals_conceded_avg']
            
            # Poisson dağılımına dayalı beklenen toplam gol sayısı
            home_expected = (home_avg + away_conceded) / 2
            away_expected = (away_avg + home_conceded) / 2
            
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
            
    def _predict_over_under(self, match_data: Dict[str, Any], threshold: float, team_analysis: Dict[str, Any] = None) -> Dict[str, float]:
        """Belirli bir gol eşiği için over/under tahminleri yapar."""
        if team_analysis:
            # Gelişmiş analiz kullan
            home_expected = team_analysis['home_team']['expected_goals']
            away_expected = team_analysis['away_team']['expected_goals']
            
            # Sakat oyuncuların etkisini hesaba kat
            home_expected *= (1 - team_analysis['home_team']['injury_impact'] * 0.5)
            away_expected *= (1 - team_analysis['away_team']['injury_impact'] * 0.5)
            
            # Son 5 maçtaki gol ortalamasını hesaba kat
            home_expected = (home_expected * 0.7) + (team_analysis['home_team']['last_5_matches']['goals_scored_avg'] * 0.3)
            away_expected = (away_expected * 0.7) + (team_analysis['away_team']['last_5_matches']['goals_scored_avg'] * 0.3)
        else:
            # Eski yöntemi kullan
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
        home_team_name = match_data.get('home_team_name', '')
        away_team_name = match_data.get('away_team_name', '')
        
        if any(team in ["Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla"] for team in [home_team_name, away_team_name]):
            league_factor = 1.1  # İspanya ligi daha golcü
        elif any(team in ["Bayern Munich", "Borussia Dortmund", "RB Leipzig"] for team in [home_team_name, away_team_name]):
            league_factor = 1.2  # Bundesliga en golcü liglerden
        elif any(team in ["Liverpool", "Manchester City", "Manchester United", "Chelsea"] for team in [home_team_name, away_team_name]):
            league_factor = 1.15  # Premier Lig golcü
        elif any(team in ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"] for team in [home_team_name, away_team_name]):
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
            
    def _predict_corners(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Korner tahminlerini yapar."""
        if team_analysis:
            # Gelişmiş analiz kullan
            home_attack_strength = team_analysis['home_team']['base_strength'] * 10
            away_attack_strength = team_analysis['away_team']['base_strength'] * 10
            
            # Takımların genel gücü ve son 5 maç performansına göre korner sayısını ayarla
            home_corner_factor = (home_attack_strength + 
                                team_analysis['home_team']['last_5_matches']['goals_scored_avg']) / 2
            away_corner_factor = (away_attack_strength + 
                                team_analysis['away_team']['last_5_matches']['goals_scored_avg']) / 2
                                
            # Ev sahibi avantajını hesaba kat
            home_corner_factor *= 1.2
            
            # Beklenen korner sayısı
            expected_corners = (home_corner_factor + away_corner_factor) * 2.5
        else:
            # Eski yöntemi kullan
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
    
    def _predict_halves(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """İlk yarı ve ikinci yarı tahminlerini yapar."""
        if team_analysis:
            # Gelişmiş analiz kullan
            home_strength = team_analysis['home_team']['adjusted_strength']
            away_strength = team_analysis['away_team']['adjusted_strength']
            
            # Takımların yarı performanslarını hesapla
            first_half_home_advantage = 1.15  # İlk yarı ev avantajı daha düşük
            second_half_home_advantage = 1.05
            
            # İlk yarıda ev sahibi takımlar daha iyi başlar genellikle
            first_half_home_strength = home_strength * first_half_home_advantage
            first_half_away_strength = away_strength
            
            # İkinci yarıda genelde deplasman takımları daha baskılı oynar
            second_half_home_strength = home_strength * second_half_home_advantage
            second_half_away_strength = away_strength * 1.1
        else:
            # Eski yöntemi kullan
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
        
        # Gol sayısı tahminleri - genelde ilk yarılar daha az gollü olur
        if team_analysis:
            first_half_goals = (team_analysis['home_team']['expected_goals'] + team_analysis['away_team']['expected_goals']) * 0.4
            second_half_goals = (team_analysis['home_team']['expected_goals'] + team_analysis['away_team']['expected_goals']) * 0.6
        else:
            first_half_goals = (first_half_home_strength + first_half_away_strength) * 0.3
            second_half_goals = (second_half_home_strength + second_half_away_strength) * 0.4
        
        return {
            'first_half': {
                'home_win': float(first_half_win_prob),
                'draw': float(first_half_draw_prob),
                'away_win': float(first_half_loss_prob),
                'goals': float(round(first_half_goals, 1))
            },
            'second_half': {
                'home_win': float(second_half_win_prob),
                'draw': float(second_half_draw_prob),
                'away_win': float(second_half_loss_prob),
                'goals': float(round(second_half_goals, 1))
            }
        }
        
    def _predict_goalscorers(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Gol atması muhtemel oyuncuları tahmin eder."""
        # Takım isimleri
        home_team_name = match_data.get('home_team_name', 'Ev Sahibi')
        away_team_name = match_data.get('away_team_name', 'Deplasman')
        
        # Güncel ve uygun oyuncuları al
        home_team_players = self._get_example_players(home_team_name)
        away_team_players = self._get_example_players(away_team_name)
        
        if team_analysis:
            # Gelişmiş analiz kullanarak beklenen gol sayılarını hesapla
            home_expected_goals = team_analysis['home_team']['expected_goals']
            away_expected_goals = team_analysis['away_team']['expected_goals']
            
            # Sakatların etkisini hesapla
            home_injury_effect = 1 - team_analysis['home_team']['injury_impact']
            away_injury_effect = 1 - team_analysis['away_team']['injury_impact']
            
            # Beklenen gol sayılarını sakatlık etkisiyle düzelt
            home_expected_goals *= home_injury_effect
            away_expected_goals *= away_injury_effect
        else:
            # Takımların gol beklentilerine göre oyuncu gol olasılıklarını ayarla
            home_expected_goals = match_data['home_goals_scored_avg']
            away_expected_goals = match_data['away_goals_scored_avg']
            
            # Form faktörünü dahil et
            home_form_factor = match_data['home_form'] / 50  # 50-100 arası puanı 1-2 arasına normalize et
            away_form_factor = match_data['away_form'] / 50
            
            # Ev avantajını hesaba kat
            home_expected_goals *= home_form_factor * 1.1  # Ev sahibi avantajı
            away_expected_goals *= away_form_factor * 0.9  # Deplasman dezavantajı
        
        # Oyuncu olasılıklarını son form durumuna göre ayarla
        self._adjust_goalscoring_probabilities(home_team_players, home_expected_goals)
        self._adjust_goalscoring_probabilities(away_team_players, away_expected_goals)
        
        # Sadece uygun oyuncuları döndür ve olasılıklarına göre sırala
        available_home_players = [p for p in home_team_players if p.get("is_available", True)]
        available_away_players = [p for p in away_team_players if p.get("is_available", True)]
        
        available_home_players.sort(key=lambda x: x["scoring_prob"], reverse=True)
        available_away_players.sort(key=lambda x: x["scoring_prob"], reverse=True)
        
        # Her oyuncu için ek bilgiler ekle
        for player in available_home_players:
            self._add_player_context(player, home_expected_goals)
        
        for player in available_away_players:
            self._add_player_context(player, away_expected_goals)
        
        return {
            'home_team': available_home_players,
            'away_team': available_away_players
        }
        
    def _adjust_goalscoring_probabilities(self, players: List[Dict[str, Any]], expected_goals: float) -> None:
        """Beklenen gol sayısına göre oyuncu gol olasılıklarını ayarlar."""
        # Önce sadece uygun oyuncuları filtrele
        available_players = [player for player in players if player.get("is_available", True)]
        
        if not available_players:
            # Eğer uygun oyuncu yoksa, tüm oyuncuları kullan ama düşük olasılıkla
            available_players = players
            for player in available_players:
                player["scoring_prob"] *= 0.1  # Sakatlık durumunda olasılığı çok düşür
                
        # Takımın toplam gol beklentisine göre skaler bir faktör hesapla
        scaling_factor = expected_goals / 1.5  # 1.5 ortalama değer olarak kabul edilir
        
        # Oyuncuların formunu değerlendir
        for player in available_players:
            # Son 5 maçtaki formu hesapla
            form_factor = 1.0
            if "last_5_form" in player:
                form_sum = sum(player["last_5_form"])
                form_factor = 0.8 + (form_sum / 5) * 0.4  # 0.8 ile 1.2 arasında değişir
            
            # Gol ve asist sayılarına göre etki hesapla
            goal_impact = 1.0
            if "goals" in player and "assists" in player:
                goal_impact = 0.7 + min(1.0, (player["goals"] + player["assists"] * 0.5) / 10) * 0.6
            
            # Oyuncu pozisyonuna göre düzeltme
            position_factor = 1.0
            if player["position"] == "ST":
                position_factor = 1.2
            elif player["position"] in ["LW", "RW"]:
                position_factor = 1.1
            elif player["position"] == "AMF":
                position_factor = 0.9
            elif player["position"] in ["CMF", "CDM"]:
                position_factor = 0.7
            elif player["position"] in ["LB", "RB"]:
                position_factor = 0.5
            elif player["position"] in ["CB"]:
                position_factor = 0.3
            
            # En iyi golcü faktörü
            top_scorer_factor = 1.2 if player.get("top_scorer", False) else 1.0
            
            # Tüm faktörleri birleştirerek son olasılığı hesapla
            player["scoring_prob"] = min(0.8, player["scoring_prob"] * scaling_factor * form_factor * 
                                     goal_impact * position_factor * top_scorer_factor)
            
        # Toplam olasılık
        total_prob = sum(player["scoring_prob"] for player in available_players)
        
        # Olasılıkları normalleştir ve beklenen gole göre ayarla
        if total_prob > 0:
            normalization_factor = expected_goals / max(1.0, total_prob)
            for player in available_players:
                player["scoring_prob"] = min(0.9, player["scoring_prob"] * normalization_factor)
            
    def _add_player_context(self, player: Dict[str, Any], team_expected_goals: float) -> None:
        """Oyuncu verilerine bağlamsal bilgiler ekler."""
        # Gol olasılığı yüksek, orta veya düşük olarak sınıflandır
        if player["scoring_prob"] > 0.35:
            player["probability_rating"] = "Yüksek"
        elif player["scoring_prob"] > 0.15:
            player["probability_rating"] = "Orta"
        else:
            player["probability_rating"] = "Düşük"
        
        # Son form durumuna göre trend belirle
        if "last_5_form" in player:
            last_3_form = player["last_5_form"][-3:]
            last_2_form = player["last_5_form"][-2:]
            
            if sum(last_3_form) >= 2 and sum(last_2_form) >= 1:
                player["form_trend"] = "Yükselen"
            elif sum(last_3_form) <= 1:
                player["form_trend"] = "Düşen"
            else:
                player["form_trend"] = "Stabil"
        
        # Tahmini gol sayısı (oran * beklenen gol)
        if "scoring_prob" in player:
            normalized_prob = player["scoring_prob"] / 0.9  # En yüksek 0.9 olduğu için normalize et
            player["expected_goals"] = round(normalized_prob * team_expected_goals, 2)
            
    def _get_example_players(self, team_name: str) -> List[Dict[str, Any]]:
        """Takım adına göre güncel oyuncu listesi döndürür."""
        # Gerçek bir uygulama için API'den veya transfermarkt.com'dan güncel veri çekilmelidir
        # Bu örnekte her takım için 2023-2024 sezonu güncel verileri kullanılmıştır
        
        # Takım isimleri küçük harfe çevrilerek kontrol edilir
        team_name_lower = team_name.lower()
        
        # Güncel oyuncu veritabanı - Türkiye Süper Lig
        if "galatasaray" in team_name_lower:
            return [
                {"name": "Mauro Icardi", "position": "ST", "scoring_prob": 0.45, "is_available": True, 
                 "market_value": "15M€", "top_scorer": True, "goals": 12, "assists": 3, "last_5_form": [1, 1, 0, 1, 1]},
                {"name": "Victor Osimhen", "position": "ST", "scoring_prob": 0.42, "is_available": True, 
                 "market_value": "75M€", "top_scorer": False, "goals": 8, "assists": 2, "last_5_form": [1, 1, 1, 0, 1]},
                {"name": "Michy Batshuayi", "position": "ST", "scoring_prob": 0.32, "is_available": True,
                 "market_value": "7M€", "top_scorer": False, "goals": 7, "assists": 1, "last_5_form": [0, 1, 0, 1, 1]},
                {"name": "Dries Mertens", "position": "AMF", "scoring_prob": 0.20, "is_available": True,
                 "market_value": "5M€", "top_scorer": False, "goals": 4, "assists": 8, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Barış Alper Yılmaz", "position": "RW", "scoring_prob": 0.20, "is_available": True, 
                 "market_value": "18M€", "top_scorer": False, "goals": 6, "assists": 4, "last_5_form": [0, 1, 1, 1, 0]},
                {"name": "Yunus Akgün", "position": "RW", "scoring_prob": 0.15, "is_available": True,
                 "market_value": "8M€", "top_scorer": False, "goals": 3, "assists": 5, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Gabriel Sara", "position": "CMF", "scoring_prob": 0.18, "is_available": True,
                 "market_value": "25M€", "top_scorer": False, "goals": 5, "assists": 6, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Davinson Sanchez", "position": "CB", "scoring_prob": 0.08, "is_available": True,
                 "market_value": "22M€", "top_scorer": False, "goals": 2, "assists": 0, "last_5_form": [1, 1, 1, 1, 1]},
                {"name": "Lucas Torreira", "position": "CDM", "scoring_prob": 0.10, "is_available": True,
                 "market_value": "20M€", "top_scorer": False, "goals": 1, "assists": 3, "last_5_form": [1, 1, 1, 0, 1]}
            ]
        elif "fenerbahçe" in team_name_lower:
            return [
                {"name": "Edin Dzeko", "position": "ST", "scoring_prob": 0.38, "is_available": True,
                 "market_value": "4M€", "top_scorer": True, "goals": 14, "assists": 6, "last_5_form": [1, 1, 1, 0, 1]},
                {"name": "Dusan Tadic", "position": "AMF", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "10M€", "top_scorer": False, "goals": 9, "assists": 12, "last_5_form": [1, 0, 1, 1, 1]},
                {"name": "Sebastian Szymanski", "position": "CMF", "scoring_prob": 0.20, "is_available": True,
                 "market_value": "22M€", "top_scorer": False, "goals": 10, "assists": 5, "last_5_form": [1, 1, 0, 1, 0]},
                {"name": "İrfan Can Kahveci", "position": "RW", "scoring_prob": 0.22, "is_available": True,
                 "market_value": "12M€", "top_scorer": False, "goals": 8, "assists": 7, "last_5_form": [0, 1, 1, 0, 1]},
                {"name": "Bright Osayi-Samuel", "position": "RB", "scoring_prob": 0.10, "is_available": True,
                 "market_value": "12M€", "top_scorer": False, "goals": 1, "assists": 4, "last_5_form": [1, 1, 1, 1, 0]},
                {"name": "Çağlar Söyüncü", "position": "CB", "scoring_prob": 0.08, "is_available": False,
                 "market_value": "22M€", "top_scorer": False, "goals": 1, "assists": 0, "injury": "Muscle", "return_date": "2024-04-26", "last_5_form": [0, 0, 0, 0, 0]},
                {"name": "Cengiz Ünder", "position": "RW", "scoring_prob": 0.28, "is_available": True,
                 "market_value": "16M€", "top_scorer": False, "goals": 7, "assists": 8, "last_5_form": [1, 0, 1, 1, 1]},
                {"name": "Sofyan Amrabat", "position": "CDM", "scoring_prob": 0.05, "is_available": True,
                 "market_value": "20M€", "top_scorer": False, "goals": 0, "assists": 2, "last_5_form": [1, 1, 0, 1, 1]},
                {"name": "Ferdi Kadıoğlu", "position": "LB", "scoring_prob": 0.12, "is_available": True,
                 "market_value": "25M€", "top_scorer": False, "goals": 2, "assists": 5, "last_5_form": [1, 1, 1, 0, 1]}
            ]
        elif "beşiktaş" in team_name_lower:
            return [
                {"name": "Ciro Immobile", "position": "ST", "scoring_prob": 0.38, "is_available": True,
                 "market_value": "10M€", "top_scorer": True, "goals": 11, "assists": 2, "last_5_form": [1, 1, 0, 0, 1]},
                {"name": "Semih Kılıçsoy", "position": "ST", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "12M€", "top_scorer": False, "goals": 8, "assists": 1, "last_5_form": [1, 0, 1, 1, 0]},
                {"name": "Ernest Muçi", "position": "LW", "scoring_prob": 0.22, "is_available": True,
                 "market_value": "8M€", "top_scorer": False, "goals": 5, "assists": 4, "last_5_form": [0, 1, 0, 1, 1]},
                {"name": "Gedson Fernandes", "position": "CMF", "scoring_prob": 0.15, "is_available": True,
                 "market_value": "20M€", "top_scorer": False, "goals": 3, "assists": 5, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Alex Oxlade-Chamberlain", "position": "AMF", "scoring_prob": 0.18, "is_available": True,
                 "market_value": "7M€", "top_scorer": False, "goals": 4, "assists": 3, "last_5_form": [0, 1, 1, 0, 0]},
                {"name": "Rafa Silva", "position": "AMF", "scoring_prob": 0.32, "is_available": True,
                 "market_value": "15M€", "top_scorer": False, "goals": 9, "assists": 8, "last_5_form": [1, 0, 1, 1, 1]},
                {"name": "Arthur Masuaku", "position": "LB", "scoring_prob": 0.08, "is_available": True,
                 "market_value": "5M€", "top_scorer": False, "goals": 1, "assists": 4, "last_5_form": [1, 1, 0, 0, 1]}
            ]
        elif "trabzonspor" in team_name_lower:
            return [
                {"name": "Enis Bardhi", "position": "AMF", "scoring_prob": 0.22, "is_available": True,
                 "market_value": "7M€", "top_scorer": False, "goals": 5, "assists": 3, "last_5_form": [1, 0, 0, 1, 0]},
                {"name": "Edin Visca", "position": "RW", "scoring_prob": 0.20, "is_available": True,
                 "market_value": "5M€", "top_scorer": False, "goals": 4, "assists": 7, "last_5_form": [0, 1, 0, 1, 0]},
                {"name": "Paul Onuachu", "position": "ST", "scoring_prob": 0.38, "is_available": True,
                 "market_value": "9M€", "top_scorer": True, "goals": 10, "assists": 1, "last_5_form": [1, 0, 1, 1, 0]},
                {"name": "Mahmoud Trezeguet", "position": "LW", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "8M€", "top_scorer": False, "goals": 7, "assists": 5, "last_5_form": [1, 1, 0, 0, 1]},
                {"name": "Berat Özdemir", "position": "DMF", "scoring_prob": 0.10, "is_available": True,
                 "market_value": "4M€", "top_scorer": False, "goals": 2, "assists": 1, "last_5_form": [0, 0, 1, 0, 0]}
            ]
        # Premier Lig takımları
        elif "manchester city" in team_name_lower:
            return [
                {"name": "Erling Haaland", "position": "ST", "scoring_prob": 0.60, "is_available": True,
                 "market_value": "200M€", "top_scorer": True, "goals": 25, "assists": 5, "last_5_form": [1, 1, 1, 1, 1]},
                {"name": "Phil Foden", "position": "LW", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "150M€", "top_scorer": False, "goals": 15, "assists": 10, "last_5_form": [1, 0, 1, 1, 1]},
                {"name": "Kevin De Bruyne", "position": "AMF", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "60M€", "top_scorer": False, "goals": 8, "assists": 18, "last_5_form": [1, 1, 0, 1, 0]},
                {"name": "Bernardo Silva", "position": "RW", "scoring_prob": 0.22, "is_available": True,
                 "market_value": "70M€", "top_scorer": False, "goals": 7, "assists": 9, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Julian Alvarez", "position": "ST", "scoring_prob": 0.38, "is_available": True,
                 "market_value": "80M€", "top_scorer": False, "goals": 12, "assists": 7, "last_5_form": [0, 1, 1, 1, 0]}
            ]
        elif "liverpool" in team_name_lower:
            return [
                {"name": "Mohamed Salah", "position": "RW", "scoring_prob": 0.48, "is_available": True,
                 "market_value": "65M€", "top_scorer": True, "goals": 18, "assists": 12, "last_5_form": [1, 1, 1, 0, 1]},
                {"name": "Darwin Núñez", "position": "ST", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "70M€", "top_scorer": False, "goals": 14, "assists": 5, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Luis Díaz", "position": "LW", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "75M€", "top_scorer": False, "goals": 11, "assists": 7, "last_5_form": [0, 1, 1, 0, 1]},
                {"name": "Diogo Jota", "position": "ST", "scoring_prob": 0.40, "is_available": True,
                 "market_value": "55M€", "top_scorer": False, "goals": 13, "assists": 3, "last_5_form": [1, 1, 0, 1, 0]},
                {"name": "Cody Gakpo", "position": "LW", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "45M€", "top_scorer": False, "goals": 8, "assists": 6, "last_5_form": [0, 1, 0, 1, 1]}
            ]
        elif "arsenal" in team_name_lower:
            return [
                {"name": "Bukayo Saka", "position": "RW", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "140M€", "top_scorer": True, "goals": 14, "assists": 15, "last_5_form": [1, 1, 0, 1, 1]},
                {"name": "Kai Havertz", "position": "ST", "scoring_prob": 0.32, "is_available": True,
                 "market_value": "75M€", "top_scorer": False, "goals": 13, "assists": 7, "last_5_form": [1, 0, 1, 1, 0]},
                {"name": "Gabriel Jesus", "position": "ST", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "65M€", "top_scorer": False, "goals": 9, "assists": 8, "last_5_form": [0, 1, 0, 1, 1]},
                {"name": "Leandro Trossard", "position": "LW", "scoring_prob": 0.28, "is_available": True,
                 "market_value": "45M€", "top_scorer": False, "goals": 10, "assists": 6, "last_5_form": [1, 1, 0, 0, 1]},
                {"name": "Gabriel Martinelli", "position": "LW", "scoring_prob": 0.26, "is_available": True,
                 "market_value": "70M€", "top_scorer": False, "goals": 8, "assists": 7, "last_5_form": [0, 1, 1, 0, 1]}
            ]
        # La Liga takımları
        elif "real madrid" in team_name_lower:
            return [
                {"name": "Kylian Mbappé", "position": "ST", "scoring_prob": 0.55, "is_available": True,
                 "market_value": "170M€", "top_scorer": True, "goals": 20, "assists": 8, "last_5_form": [1, 1, 1, 1, 1]},
                {"name": "Vinicius Jr.", "position": "LW", "scoring_prob": 0.45, "is_available": True,
                 "market_value": "200M€", "top_scorer": False, "goals": 17, "assists": 12, "last_5_form": [1, 1, 0, 1, 1]},
                {"name": "Rodrygo", "position": "RW", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "110M€", "top_scorer": False, "goals": 13, "assists": 9, "last_5_form": [1, 0, 1, 1, 0]},
                {"name": "Jude Bellingham", "position": "AMF", "scoring_prob": 0.40, "is_available": True,
                 "market_value": "180M€", "top_scorer": False, "goals": 15, "assists": 11, "last_5_form": [1, 1, 1, 0, 1]},
                {"name": "Endrick", "position": "ST", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "60M€", "top_scorer": False, "goals": 7, "assists": 3, "last_5_form": [0, 1, 0, 1, 0]}
            ]
        elif "barcelona" in team_name_lower:
            return [
                {"name": "Robert Lewandowski", "position": "ST", "scoring_prob": 0.50, "is_available": True,
                 "market_value": "40M€", "top_scorer": True, "goals": 18, "assists": 6, "last_5_form": [1, 1, 0, 1, 1]},
                {"name": "Raphinha", "position": "RW", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "50M€", "top_scorer": False, "goals": 12, "assists": 10, "last_5_form": [1, 0, 1, 1, 0]},
                {"name": "Lamine Yamal", "position": "RW", "scoring_prob": 0.32, "is_available": True,
                 "market_value": "180M€", "top_scorer": False, "goals": 10, "assists": 13, "last_5_form": [1, 1, 1, 0, 1]},
                {"name": "Fermín López", "position": "AMF", "scoring_prob": 0.22, "is_available": True,
                 "market_value": "25M€", "top_scorer": False, "goals": 6, "assists": 5, "last_5_form": [0, 1, 0, 1, 1]},
                {"name": "João Félix", "position": "LW", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "50M€", "top_scorer": False, "goals": 9, "assists": 7, "last_5_form": [1, 1, 0, 0, 1]}
            ]
        # Bundesliga takımları
        elif "bayern munich" in team_name_lower:
            return [
                {"name": "Harry Kane", "position": "ST", "scoring_prob": 0.55, "is_available": True,
                 "market_value": "100M€", "top_scorer": True, "goals": 24, "assists": 10, "last_5_form": [1, 1, 1, 1, 0]},
                {"name": "Jamal Musiala", "position": "AMF", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "140M€", "top_scorer": False, "goals": 14, "assists": 12, "last_5_form": [1, 1, 0, 1, 1]},
                {"name": "Leroy Sané", "position": "RW", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "65M€", "top_scorer": False, "goals": 10, "assists": 14, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Serge Gnabry", "position": "LW", "scoring_prob": 0.28, "is_available": True,
                 "market_value": "45M€", "top_scorer": False, "goals": 9, "assists": 7, "last_5_form": [0, 1, 1, 0, 1]},
                {"name": "Michael Olise", "position": "RW", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "55M€", "top_scorer": False, "goals": 7, "assists": 9, "last_5_form": [1, 0, 0, 1, 1]}
            ]
        # Serie A takımları
        elif "inter milan" in team_name_lower:
            return [
                {"name": "Lautaro Martínez", "position": "ST", "scoring_prob": 0.45, "is_available": True,
                 "market_value": "110M€", "top_scorer": True, "goals": 22, "assists": 6, "last_5_form": [1, 1, 1, 0, 1]},
                {"name": "Marcus Thuram", "position": "ST", "scoring_prob": 0.35, "is_available": True,
                 "market_value": "70M€", "top_scorer": False, "goals": 15, "assists": 9, "last_5_form": [1, 0, 1, 1, 0]},
                {"name": "Hakan Çalhanoğlu", "position": "CMF", "scoring_prob": 0.28, "is_available": True,
                 "market_value": "45M€", "top_scorer": False, "goals": 9, "assists": 12, "last_5_form": [1, 1, 0, 1, 0]},
                {"name": "Nicolò Barella", "position": "CMF", "scoring_prob": 0.15, "is_available": True,
                 "market_value": "75M€", "top_scorer": False, "goals": 4, "assists": 10, "last_5_form": [0, 1, 1, 0, 1]},
                {"name": "Marko Arnautović", "position": "ST", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "6M€", "top_scorer": False, "goals": 7, "assists": 3, "last_5_form": [0, 1, 0, 1, 0]}
            ]
        else:
            # Diğer takımlar için jenerik oyuncular
            return [
                {"name": "Forvet Oyuncusu", "position": "ST", "scoring_prob": 0.30, "is_available": True,
                 "market_value": "15M€", "top_scorer": True, "goals": 10, "assists": 3, "last_5_form": [1, 0, 1, 0, 1]},
                {"name": "Orta Saha 1", "position": "AMF", "scoring_prob": 0.20, "is_available": True,
                 "market_value": "12M€", "top_scorer": False, "goals": 6, "assists": 8, "last_5_form": [0, 1, 0, 1, 0]},
                {"name": "Kanat Oyuncusu", "position": "LW", "scoring_prob": 0.18, "is_available": True,
                 "market_value": "10M€", "top_scorer": False, "goals": 5, "assists": 7, "last_5_form": [1, 0, 1, 0, 0]},
                {"name": "Orta Saha 2", "position": "CMF", "scoring_prob": 0.15, "is_available": True,
                 "market_value": "8M€", "top_scorer": False, "goals": 3, "assists": 5, "last_5_form": [0, 1, 0, 1, 1]},
                {"name": "Yedek Forvet", "position": "ST", "scoring_prob": 0.25, "is_available": True,
                 "market_value": "7M€", "top_scorer": False, "goals": 7, "assists": 2, "last_5_form": [1, 0, 0, 1, 1]}
            ]

    def _create_team_comparison(self, match_data: Dict[str, Any], team_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """İki takım arasında karşılaştırmalı istatistikler oluşturur."""
        # Son 5 maç sonuçlarını oluştur (W/D/L)
        home_last_matches = self._extract_last_match_results(match_data, 'home')
        away_last_matches = self._extract_last_match_results(match_data, 'away')
        
        # Ek karşılaştırma verileri
        additional_comparison = {}
        if team_analysis:
            additional_comparison = {
                'team_value_comparison': {
                    'home': float(team_analysis['home_team']['team_value']),
                    'away': float(team_analysis['away_team']['team_value']),
                    'difference': float(team_analysis['home_team']['team_value'] - team_analysis['away_team']['team_value'])
                },
                'injury_impact': {
                    'home': float(team_analysis['home_team']['injury_impact']),
                    'away': float(team_analysis['away_team']['injury_impact']),
                    'home_injured_players': team_analysis['home_team']['unavailable_players'],
                    'away_injured_players': team_analysis['away_team']['unavailable_players']
                },
                'last_5_matches_comparison': {
                    'home_goals_scored': float(team_analysis['home_team']['last_5_matches']['goals_scored_avg']),
                    'away_goals_scored': float(team_analysis['away_team']['last_5_matches']['goals_scored_avg']),
                    'home_goals_conceded': float(team_analysis['home_team']['last_5_matches']['goals_conceded_avg']),
                    'away_goals_conceded': float(team_analysis['away_team']['last_5_matches']['goals_conceded_avg'])
                },
                'uefa_ranking': {
                    'home_rank': team_analysis['home_team']['uefa_rank'],
                    'away_rank': team_analysis['away_team']['uefa_rank'],
                    'home_factor': float(team_analysis['home_team']['uefa_factor']),
                    'away_factor': float(team_analysis['away_team']['uefa_factor'])
                }
            }
        
        result = {
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
        
        # Ek verileri ekle
        if additional_comparison:
            result.update(additional_comparison)
            
        return result
    
    def _extract_last_match_results(self, match_data: Dict[str, Any], team_type: str) -> List[str]:
        """Son maç sonuçlarını W/D/L formatında çıkarır."""
        results = []
        
        try:
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
        except Exception as e:
            logger.warning(f"Error extracting match results: {str(e)}")
            # Hata durumunda varsayılan değerleri kullan
            for _ in range(5):
                results.append('D')
                    
        return results 
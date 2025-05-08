import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Any, Tuple
import pandas as pd
import joblib
import os
from datetime import datetime


class FootballMLModel:
    """Futbol maç sonuçlarını tahmin eden derin öğrenme modeli."""
    
    def __init__(self, model_dir: str = "models"):
        """ML modelini başlatır.
        
        Args:
            model_dir: Model dosyalarının saklanacağı dizin
        """
        self.model_dir = model_dir
        self.team_model = None
        self.player_model = None
        self.combined_model = None
        self.team_scaler = None
        self.player_scaler = None
        self.label_encoders = {}
        
        # Modelin varlığını kontrol et
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.team_model_path = os.path.join(model_dir, "team_model.h5")
        self.player_model_path = os.path.join(model_dir, "player_model.h5")
        self.combined_model_path = os.path.join(model_dir, "combined_model.h5")
        self.scaler_path = os.path.join(model_dir, "scalers.joblib")
        self.encoder_path = os.path.join(model_dir, "encoders.joblib")
        
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Modelleri yükler veya oluşturur."""
        try:
            if os.path.exists(self.team_model_path) and os.path.exists(self.player_model_path) and os.path.exists(self.combined_model_path):
                print("Kayıtlı modeller yükleniyor...")
                self.team_model = tf.keras.models.load_model(self.team_model_path)
                self.player_model = tf.keras.models.load_model(self.player_model_path)
                self.combined_model = tf.keras.models.load_model(self.combined_model_path)
                
                # Scaler ve encoderları yükle
                if os.path.exists(self.scaler_path):
                    scalers = joblib.load(self.scaler_path)
                    self.team_scaler = scalers.get("team_scaler")
                    self.player_scaler = scalers.get("player_scaler")
                
                if os.path.exists(self.encoder_path):
                    self.label_encoders = joblib.load(self.encoder_path)
                
                print("Modeller başarıyla yüklendi.")
            else:
                print("Kayıtlı model bulunamadı. Yeni model oluşturuluyor...")
                self._create_models()
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            print("Yeni model oluşturuluyor...")
            self._create_models()
    
    def _create_models(self):
        """Yeni modeller oluşturur."""
        # Takım tabanlı model
        self._create_team_model()
        
        # Oyuncu tabanlı model
        self._create_player_model()
        
        # Birleşik model
        self._create_combined_model()
    
    def _create_team_model(self):
        """Takım istatistikleri temelli model oluşturur."""
        # Girdi: Takım istatistikleri (form, puan, gol, xG, vb.)
        input_dim = 20  # Örnek: 20 takım özelliği
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 çıktı: Ev sahibi kazanır, Beraberlik, Deplasman kazanır
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.team_model = model
        model.save(self.team_model_path)
    
    def _create_player_model(self):
        """Oyuncu istatistikleri temelli model oluşturur."""
        # Girdi: Oyuncu özellikleri (gol, asist, form, vb.)
        input_dim = 15  # Örnek: 15 oyuncu özelliği
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')  # 4 çıktı: 0-1 gol, 2-3 gol, 4-5 gol, 6+ gol
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.player_model = model
        model.save(self.player_model_path)
    
    def _create_combined_model(self):
        """Takım ve oyuncu modellerini birleştiren kombine model oluşturur."""
        # Takım modeli girdisi
        team_input = Input(shape=(20,))
        team_features = self.team_model.layers[0](team_input)
        team_features = self.team_model.layers[1](team_features)
        team_features = self.team_model.layers[2](team_features)
        team_features = self.team_model.layers[3](team_features)
        team_features = self.team_model.layers[4](team_features)
        
        # Oyuncu modeli girdisi
        player_input = Input(shape=(15,))
        player_features = self.player_model.layers[0](player_input)
        player_features = self.player_model.layers[1](player_features)
        player_features = self.player_model.layers[2](player_features)
        player_features = self.player_model.layers[3](player_features)
        player_features = self.player_model.layers[4](player_features)
        
        # Özellikleri birleştir
        combined = concatenate([team_features, player_features])
        
        # Birleştirilmiş özellikler üzerinde ek katmanlar
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        
        # Çıktı katmanları
        match_result = Dense(3, activation='softmax', name='match_result')(x)  # Maç sonucu
        goals = Dense(10, activation='softmax', name='goals')(x)  # Toplam gol sayısı (0-9+)
        score_line = Dense(25, activation='softmax', name='score_line')(x)  # Skor tahmini (0-0 ile 4-4 arası 25 farklı skor)
        
        # Kombine model
        model = Model(
            inputs=[team_input, player_input],
            outputs=[match_result, goals, score_line]
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss={
                'match_result': 'categorical_crossentropy',
                'goals': 'categorical_crossentropy',
                'score_line': 'categorical_crossentropy'
            },
            metrics={
                'match_result': 'accuracy',
                'goals': 'accuracy',
                'score_line': 'accuracy'
            },
            loss_weights={
                'match_result': 1.0,
                'goals': 0.8,
                'score_line': 0.6
            }
        )
        
        self.combined_model = model
        model.save(self.combined_model_path)
    
    def train(self, 
              team_data: List[Dict[str, Any]], 
              player_data: List[Dict[str, Any]], 
              match_results: List[Dict[str, Any]],
              epochs: int = 50, 
              batch_size: int = 32,
              validation_split: float = 0.2):
        """Modeli eğitir.
        
        Args:
            team_data: Takım özellikleri listesi
            player_data: Oyuncu özellikleri listesi
            match_results: Maç sonuçları
            epochs: Eğitim epoch sayısı
            batch_size: Mini-batch boyutu
            validation_split: Doğrulama veri seti oranı
        """
        print("Veri ön işleme başlatılıyor...")
        # Verileri ön işle
        X_team, X_player, y_result, y_goals, y_score = self._preprocess_data(team_data, player_data, match_results)
        
        print(f"Eğitim başlatılıyor... Veri boyutu: {len(X_team)} örnek")
        
        # Takım modelini eğit
        self.team_model.fit(
            X_team, y_result,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Oyuncu modelini eğit
        self.player_model.fit(
            X_player, y_goals,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Kombine modeli eğit
        self.combined_model.fit(
            [X_team, X_player],
            [y_result, y_goals, y_score],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Modelleri kaydet
        self._save_models()
        
        print("Model eğitimi tamamlandı.")
    
    def _preprocess_data(self, 
                         team_data: List[Dict[str, Any]], 
                         player_data: List[Dict[str, Any]], 
                         match_results: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Eğitim verilerini ön işler.
        
        Args:
            team_data: Takım özellikleri listesi
            player_data: Oyuncu özellikleri listesi
            match_results: Maç sonuçları
            
        Returns:
            X_team: Takım verileri
            X_player: Oyuncu verileri
            y_result: Maç sonuçları (one-hot encoding)
            y_goals: Toplam gol sayıları (one-hot encoding)
            y_score: Skor çizgileri (one-hot encoding)
        """
        # Veri çerçeveleri oluştur
        team_df = pd.DataFrame(team_data)
        player_df = pd.DataFrame(player_data)
        results_df = pd.DataFrame(match_results)
        
        # Eksik verileri doldur
        team_df.fillna(0, inplace=True)
        player_df.fillna(0, inplace=True)
        results_df.fillna(0, inplace=True)
        
        # Kategorik değişkenleri kodla
        for col in team_df.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoders[col] = LabelEncoder()
            
            team_df[col] = self.label_encoders[col].fit_transform(team_df[col])
        
        for col in player_df.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoders[col] = LabelEncoder()
            
            player_df[col] = self.label_encoders[col].fit_transform(player_df[col])
        
        # Verileri normalize et
        if self.team_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.team_scaler = StandardScaler()
            self.player_scaler = StandardScaler()
            
            X_team = self.team_scaler.fit_transform(team_df)
            X_player = self.player_scaler.fit_transform(player_df)
        else:
            X_team = self.team_scaler.transform(team_df)
            X_player = self.player_scaler.transform(player_df)
        
        # Hedef değişkenleri one-hot kodla
        from tensorflow.keras.utils import to_categorical
        
        # Maç sonuçları (0: Ev sahibi kazanır, 1: Beraberlik, 2: Deplasman kazanır)
        results = results_df['result'].values
        y_result = to_categorical(results, num_classes=3)
        
        # Toplam gol sayıları (0-9+)
        total_goals = np.minimum(results_df['total_goals'].values, 9)  # 9'dan büyük değerleri 9 olarak sınırla
        y_goals = to_categorical(total_goals, num_classes=10)
        
        # Skor çizgileri (0-0 ile 4-4 arası, 25 farklı skor)
        # Skoru 0-24 arası bir indekse dönüştür: 0-0=0, 1-0=1, 0-1=2, 2-0=3, vb.
        home_goals = np.minimum(results_df['home_goals'].values, 4)
        away_goals = np.minimum(results_df['away_goals'].values, 4)
        score_indices = home_goals * 5 + away_goals
        y_score = to_categorical(score_indices, num_classes=25)
        
        return X_team, X_player, y_result, y_goals, y_score
    
    def predict(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Maç sonucunu tahmin eder.
        
        Args:
            match_data: Maç verileri (takım ve oyuncu bilgilerini içerir)
            
        Returns:
            Tahmin sonuçları
        """
        # Takım ve oyuncu verilerini ayır
        team_features = self._extract_team_features(match_data)
        player_features = self._extract_player_features(match_data)
        
        # Verileri ön işle
        X_team = self.team_scaler.transform([team_features]) if self.team_scaler else np.array([team_features])
        X_player = self.player_scaler.transform([player_features]) if self.player_scaler else np.array([player_features])
        
        # Tahmin yap
        match_result_probs, total_goals_probs, score_line_probs = self.combined_model.predict([X_team, X_player])
        
        # En yüksek olasılıklı sonuçları bul
        result_index = np.argmax(match_result_probs[0])
        goals_index = np.argmax(total_goals_probs[0])
        score_index = np.argmax(score_line_probs[0])
        
        # Sonuç etiketleri
        result_labels = ['HOME_WIN', 'DRAW', 'AWAY_WIN']
        
        # Skor indeksini ev sahibi ve deplasman gollerine dönüştür
        home_goals = score_index // 5
        away_goals = score_index % 5
        
        # Tahmin sonuçlarını döndür
        return {
            "match_result": {
                "prediction": result_labels[result_index],
                "home_win_prob": float(match_result_probs[0][0]),
                "draw_prob": float(match_result_probs[0][1]),
                "away_win_prob": float(match_result_probs[0][2])
            },
            "goals": {
                "prediction": int(goals_index),
                "probabilities": {f"{i}": float(total_goals_probs[0][i]) for i in range(10)}
            },
            "score": {
                "prediction": f"{home_goals}-{away_goals}",
                "probability": float(score_line_probs[0][score_index]),
                "top_scores": self._get_top_scores(score_line_probs[0])
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_team_features(self, match_data: Dict[str, Any]) -> List[float]:
        """Maç verilerinden takım özelliklerini çıkarır.
        
        Args:
            match_data: Maç verileri
            
        Returns:
            Takım özellikleri listesi
        """
        # Örnek takım özellikleri (gerçek uygulamada daha kapsamlı olabilir)
        features = []
        
        # Ev sahibi takım özellikleri
        home_team = match_data.get("home_team", {})
        features.extend([
            home_team.get("form_points", 0) / 15,  # Son 5 maçtaki puan (0-15 arası)
            home_team.get("goals_scored", 0) / 10,  # Son 5 maçtaki atılan goller
            home_team.get("goals_conceded", 0) / 10,  # Son 5 maçtaki yenilen goller
            home_team.get("xG", 0) / 10,  # Beklenen goller (xG)
            home_team.get("xGA", 0) / 10,  # Beklenen yenilen goller (xGA)
            home_team.get("possession", 50) / 100,  # Top hakimiyeti yüzdesi
            home_team.get("shots", 0) / 20,  # Şut sayısı
            home_team.get("shots_on_target", 0) / 10,  # İsabetli şut
            home_team.get("corners", 0) / 10,  # Kornerler
            home_team.get("home_advantage", 0.6),  # Ev sahibi avantajı
        ])
        
        # Deplasman takımı özellikleri
        away_team = match_data.get("away_team", {})
        features.extend([
            away_team.get("form_points", 0) / 15,
            away_team.get("goals_scored", 0) / 10,
            away_team.get("goals_conceded", 0) / 10,
            away_team.get("xG", 0) / 10,
            away_team.get("xGA", 0) / 10,
            away_team.get("possession", 50) / 100,
            away_team.get("shots", 0) / 20,
            away_team.get("shots_on_target", 0) / 10,
            away_team.get("corners", 0) / 10,
            away_team.get("away_disadvantage", 0.4),  # Deplasman dezavantajı
        ])
        
        return features
    
    def _extract_player_features(self, match_data: Dict[str, Any]) -> List[float]:
        """Maç verilerinden oyuncu özelliklerini çıkarır.
        
        Args:
            match_data: Maç verileri
            
        Returns:
            Oyuncu özellikleri listesi
        """
        # Örnek oyuncu özellikleri (gerçek uygulamada daha kapsamlı olabilir)
        features = []
        
        # Ev sahibi takım oyuncuları
        home_players = match_data.get("home_players", [])
        
        # Ev sahibi forvet oyuncuları
        home_forwards = [p for p in home_players if p.get("position") in ["ST", "CF", "LW", "RW"]]
        home_forward_goals = sum(p.get("goals", 0) for p in home_forwards) / 50  # Normalleştir
        home_forward_shots = sum(p.get("shots_per_game", 0) for p in home_forwards) / 10
        
        # Ev sahibi orta saha oyuncuları
        home_midfielders = [p for p in home_players if p.get("position") in ["AMF", "CMF", "DMF", "LMF", "RMF"]]
        home_mid_assists = sum(p.get("assists", 0) for p in home_midfielders) / 30
        home_mid_key_passes = sum(p.get("key_passes", 0) for p in home_midfielders) / 20
        
        # Ev sahibi defans oyuncuları
        home_defenders = [p for p in home_players if p.get("position") in ["CB", "LB", "RB", "LWB", "RWB"]]
        home_def_strength = len(home_defenders) * sum(p.get("rating", 7) for p in home_defenders) / 100
        
        features.extend([
            home_forward_goals,
            home_forward_shots,
            home_mid_assists,
            home_mid_key_passes,
            home_def_strength,
            len(home_forwards) / 5,
            sum(1 for p in home_players if p.get("is_available", True)) / 11,  # Sakatlık durumu
        ])
        
        # Deplasman takımı oyuncuları
        away_players = match_data.get("away_players", [])
        
        # Deplasman forvet oyuncuları
        away_forwards = [p for p in away_players if p.get("position") in ["ST", "CF", "LW", "RW"]]
        away_forward_goals = sum(p.get("goals", 0) for p in away_forwards) / 50
        away_forward_shots = sum(p.get("shots_per_game", 0) for p in away_forwards) / 10
        
        # Deplasman orta saha oyuncuları
        away_midfielders = [p for p in away_players if p.get("position") in ["AMF", "CMF", "DMF", "LMF", "RMF"]]
        away_mid_assists = sum(p.get("assists", 0) for p in away_midfielders) / 30
        away_mid_key_passes = sum(p.get("key_passes", 0) for p in away_midfielders) / 20
        
        # Deplasman defans oyuncuları
        away_defenders = [p for p in away_players if p.get("position") in ["CB", "LB", "RB", "LWB", "RWB"]]
        away_def_strength = len(away_defenders) * sum(p.get("rating", 7) for p in away_defenders) / 100
        
        features.extend([
            away_forward_goals,
            away_forward_shots,
            away_mid_assists,
            away_mid_key_passes,
            away_def_strength,
            len(away_forwards) / 5,
            sum(1 for p in away_players if p.get("is_available", True)) / 11,
        ])
        
        return features
    
    def _get_top_scores(self, score_probs: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """En yüksek olasılıklı skorları döndürür.
        
        Args:
            score_probs: Skor olasılıkları
            top_n: Döndürülecek en olası sonuç sayısı
            
        Returns:
            En olası skorlar listesi
        """
        top_indices = np.argsort(score_probs)[-top_n:][::-1]
        
        top_scores = []
        for idx in top_indices:
            home_goals = idx // 5
            away_goals = idx % 5
            top_scores.append({
                "score": f"{home_goals}-{away_goals}",
                "probability": float(score_probs[idx])
            })
        
        return top_scores
    
    def _save_models(self):
        """Modelleri ve ilgili nesneleri (scaler, encoder vb.) kaydeder."""
        # Modelleri kaydet
        self.team_model.save(self.team_model_path)
        self.player_model.save(self.player_model_path)
        self.combined_model.save(self.combined_model_path)
        
        # Scaler ve encoderları kaydet
        scalers = {
            "team_scaler": self.team_scaler,
            "player_scaler": self.player_scaler
        }
        joblib.dump(scalers, self.scaler_path)
        joblib.dump(self.label_encoders, self.encoder_path)
        
        print(f"Modeller {self.model_dir} dizinine kaydedildi.") 
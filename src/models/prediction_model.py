import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
from typing import Dict, Any
import logging
from sklearn.preprocessing import StandardScaler
import joblib
import os

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
            # Veriyi modele uygun formata dönüştür
            values_list = list(match_data.values())
            features = np.array([values_list])
            
            # Tahmin yap - eager execution kullanarak
            with tf.device('/CPU:0'):  # CPU kullanarak daha kararlı çalışır
                predictions = self.model(features, training=False).numpy()
            
            # Tahminleri yorumla
            win_prob, draw_prob, loss_prob, over_prob, under_prob, btts_yes_prob, btts_no_prob = predictions[0]
            
            # Skor tahmini
            expected_goals_home = match_data['home_goals_scored_avg'] * 1.1  # Ev sahibi avantajı
            expected_goals_away = match_data['away_goals_scored_avg'] * 0.9  # Deplasman dezavantajı
            
            return {
                'win_prob': float(win_prob),
                'draw_prob': float(draw_prob),
                'loss_prob': float(loss_prob),
                'over_under_2_5': {
                    'over': float(over_prob),
                    'under': float(under_prob)
                },
                'btts_prediction': {
                    'yes': float(btts_yes_prob),
                    'no': float(btts_no_prob)
                },
                'score_prediction': {
                    'home': round(expected_goals_home),
                    'away': round(expected_goals_away)
                },
                'corner_prediction': self._predict_corners(match_data)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
            
    def _predict_corners(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Korner tahminlerini yapar."""
        # Basit bir korner tahmini algoritması
        home_attack_strength = match_data['home_goals_scored_avg']
        away_attack_strength = match_data['away_goals_scored_avg']
        
        expected_corners = (home_attack_strength + away_attack_strength) * 5
        
        return {
            'total_corners': round(expected_corners),
            'over_8.5': float(expected_corners > 8.5),
            'over_9.5': float(expected_corners > 9.5),
            'over_10.5': float(expected_corners > 10.5)
        } 
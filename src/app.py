import os
# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from models.prediction_model import PredictionModel
from data.data_collector import DataCollector
from data.data_processor import DataProcessor
import signal
import sys
import json
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import traceback

app = Flask(__name__, template_folder='../templates', static_folder='../static')
load_dotenv()

# Model ve veri işleyicileri başlat
data_collector = DataCollector()
data_processor = DataProcessor()
prediction_model = PredictionModel()

def signal_handler(sig, frame):
    print('\nUygulama kapatılıyor...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Gerekli verileri topla
        home_team = data['home_team']
        away_team = data['away_team']
        
        # Takım verilerini topla
        home_team_data = data_collector.get_team_stats(home_team)
        away_team_data = data_collector.get_team_stats(away_team)
        
        # Verileri işle
        processed_data = data_processor.process_match_data(home_team_data, away_team_data)
        
        # Takım isimlerini ekle - tahmin modeli için
        processed_data['home_team_name'] = home_team
        processed_data['away_team_name'] = away_team
        
        # Tahminleri yap
        predictions = prediction_model.predict(processed_data)
        
        return jsonify({
            'success': True,
            'predictions': {
                'match_result': {
                    'home_win': predictions['match_result']['home_win'],
                    'draw': predictions['match_result']['draw'],
                    'away_win': predictions['match_result']['away_win']
                },
                'score_prediction': predictions['correct_score'],
                'total_goals': predictions['total_goals'],
                'over_under_goals': predictions['over_under'],
                'btts_prediction': {
                    'yes': float(predictions.get('btts_probability', 0.5)),
                    'no': float(1 - predictions.get('btts_probability', 0.5))
                },
                'half_predictions': predictions['halves'],
                'corner_prediction': predictions['corners'],
                'goalscorer_predictions': predictions['goalscorers']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Tahmin modeli ayarlarını görüntüler veya değiştirir."""
    if request.method == 'GET':
        # Mevcut ayarları getir
        current_settings = {
            "hybrid_mode": prediction_model.get_hybrid_mode(),
            "api_status": {
                "key_configured": bool(prediction_model.api_client.api_key),
                "base_url": prediction_model.api_client.base_url
            },
            "cache": {
                "duration": prediction_model.api_client.cache_duration,
                "size": len(prediction_model.api_client.cache)
            }
        }
        return jsonify({
            'success': True,
            'settings': current_settings
        })
    else:
        try:
            data = request.get_json()
            
            # Hibrit mod ayarını güncelle
            if 'hybrid_mode' in data:
                hybrid_value = float(data['hybrid_mode'])
                prediction_model.set_hybrid_mode(hybrid_value)
                
            # Önbellek süresini güncelle (isteğe bağlı)
            if 'cache_duration' in data:
                prediction_model.api_client.cache_duration = int(data['cache_duration'])
            
            # Önbelleği temizle (isteğe bağlı)
            if data.get('clear_cache', False):
                prediction_model.api_client.cache = {}
            
            return jsonify({
                'success': True,
                'message': 'Ayarlar güncellendi',
                'current_settings': {
                    'hybrid_mode': prediction_model.get_hybrid_mode()
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400

@app.route('/train', methods=['POST'])
def train_model():
    """ML modelini eğitir."""
    try:
        data = request.get_json()
        
        # Eğitim verilerini al
        team_data = data.get('team_data', [])
        player_data = data.get('player_data', [])
        match_results = data.get('match_results', [])
        
        # Opsiyonel parametreler
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        validation_split = data.get('validation_split', 0.2)
        
        # Yeterli veri var mı kontrol et
        if len(match_results) < 10:
            return jsonify({
                'success': False,
                'error': 'Eğitim için en az 10 maç sonucu gereklidir.'
            }), 400
            
        # Modeli eğit
        print(f"Model eğitimi başlatıldı: {len(match_results)} maç, {epochs} epoch")
        prediction_model.ml_model.train(
            team_data=team_data,
            player_data=player_data,
            match_results=match_results,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        return jsonify({
            'success': True,
            'message': f'Model başarıyla eğitildi: {len(match_results)} maç kullanıldı',
            'training_info': {
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print('Tahmin uygulaması başlatılıyor...')
    print('Ctrl+C ile kapatabilirsiniz.')
    
    # Debug mesajları
    print('Debug: Model ve veri servisleri yükleniyor...')
    try:
        print('Debug: PredictionModel başlatılıyor...')
        test_prediction = prediction_model.predict({
            'home_team_name': 'Test Takım 1',
            'away_team_name': 'Test Takım 2',
            'home_form': 70,
            'away_form': 60,
            'home_season_points': 30,
            'away_season_points': 25,
            'home_goals_scored_avg': 1.5,
            'away_goals_scored_avg': 1.2,
            'home_goals_conceded_avg': 0.8,
            'away_goals_conceded_avg': 1.0
        })
        print('Debug: Test tahmin başarılı!')
        print(f"Debug: Tahmin anahtarları: {list(test_prediction.keys())}")
        print(f"Debug: Maç sonucu anahtarları: {list(test_prediction['match_result'].keys()) if 'match_result' in test_prediction else 'match_result anahtarı bulunamadı'}")
    except Exception as e:
        print(f'Debug Hatası: {str(e)}')
        traceback_str = traceback.format_exc()
        print(f'Debug Hata Ayrıntıları: {traceback_str}')
    
    print('Debug: Flask uygulaması başlatılıyor...')
    app.run(debug=True, use_reloader=False)  # use_reloader=False to prevent double initialization 
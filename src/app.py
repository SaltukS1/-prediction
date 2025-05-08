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
                    'win_probability': predictions['win_prob'],
                    'draw_probability': predictions['draw_prob'],
                    'loss_probability': predictions['loss_prob']
                },
                'score_prediction': predictions['score_prediction'],
                'total_goals': predictions['total_goals'],
                'over_under_goals': predictions['over_under_goals'],
                'btts_prediction': predictions['btts_prediction'],
                'half_predictions': predictions['half_predictions'],
                'corner_prediction': predictions['corner_prediction'],
                'goalscorer_predictions': predictions['goalscorer_predictions']
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
    app.run(debug=True, use_reloader=False)  # use_reloader=False to prevent double initialization 
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from models.prediction_model import PredictionModel
from data.data_collector import DataCollector
from data.data_processor import DataProcessor

app = Flask(__name__, template_folder='../templates', static_folder='../static')
load_dotenv()

# Model ve veri işleyicileri başlat
data_collector = DataCollector()
data_processor = DataProcessor()
prediction_model = PredictionModel()

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
        
        # Tahminleri yap
        predictions = prediction_model.predict(processed_data)
        
        return jsonify({
            'success': True,
            'predictions': {
                'win_probability': predictions['win_prob'],
                'draw_probability': predictions['draw_prob'],
                'loss_probability': predictions['loss_prob'],
                'over_under_2_5': predictions['over_under_2_5'],
                'corner_prediction': predictions['corner_prediction'],
                'btts_prediction': predictions['btts_prediction'],
                'score_prediction': predictions['score_prediction']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 
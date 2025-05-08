# Futbol Maç Tahmin Sistemi

Bu proje, güncel futbol verilerini API'den çeken ve makine öğrenmesi ile istatistiksel yöntemleri birleştirerek maç sonuçlarını tahmin eden gelişmiş bir sistemdir.

## Özellikler

- **Gerçek Zamanlı Veri**: Football-data.org API entegrasyonu ile güncel takım ve oyuncu verileri
- **Derin Öğrenme Modeli**: TensorFlow tabanlı derin öğrenme ile maç sonuçları tahmin sistemi
- **Hibrit Tahmin Motoru**: İstatistiksel analiz ve makine öğrenmesi tahminlerini birleştiren hibrit sistem
- **Dinamik Oyuncu Analizi**: Güncel oyuncu formlarını ve sakatlık durumlarını dikkate alan analiz
- **Gelişmiş İstatistikler**: Gol sayısı, skor tahmini, korner, ilk/ikinci yarı golü, golcü tahmini
- **Web Arayüzü**: Flask tabanlı kullanıcı dostu web arayüzü

## Kurulum

1. Gereksinimleri yükleyin:
```
pip install -r requirements.txt
```

2. API anahtarınızı ayarlayın:
```
# Windows
set FOOTBALL_API_KEY=your_api_key_here

# Linux/macOS
export FOOTBALL_API_KEY=your_api_key_here
```

3. Uygulamayı başlatın:
```
python src/app.py
```

## Kullanım

### Web Arayüzü

Tarayıcıdan `http://localhost:5000` adresine giderek web arayüzünü kullanabilirsiniz.

### API ile Tahmin

```python
import requests
import json

# Tahmin isteği
response = requests.post('http://localhost:5000/predict', json={
    'home_team': 'Galatasaray',
    'away_team': 'Fenerbahçe',
    'league': 'Süper Lig',
    'match_date': '2023-10-22'
})

# Sonucu görüntüle
prediction = response.json()
print(json.dumps(prediction, indent=4))
```

## Modelin Eğitilmesi

ML modelini mevcut verilerle eğitmek için:

```python
import requests

# Eğitim verileri
training_data = {
    'team_data': [...],  # Takım verileri
    'player_data': [...],  # Oyuncu verileri
    'match_results': [...],  # Maç sonuçları
    'epochs': 50,
    'batch_size': 32
}

# Modeli eğit
response = requests.post('http://localhost:5000/train', json=training_data)
print(response.json())
```

## Hibrit Mod Ayarı

Hibrit mod, istatistiksel ve ML tahminlerinin ağırlıklarını belirler:

- 0.0: Sadece istatistiksel tahmin
- 1.0: Sadece ML tahmini
- 0.5: Dengeli karışım (varsayılan)

```python
import requests

# Hibrit modu değiştir
response = requests.post('http://localhost:5000/settings', json={
    'hybrid_mode': 0.7  # ML modeline daha fazla ağırlık ver
})
print(response.json())
```

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasını inceleyebilirsiniz.

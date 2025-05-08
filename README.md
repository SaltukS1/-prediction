# Futbol Maç Tahmin Uygulaması

Bu uygulama, futbol maçlarının sonuçlarını tahmin etmek için makine öğrenimi ve istatistiksel modeller kullanan bir web uygulamasıdır.

## Özellikler

- Maç sonuçları tahmini (Kazanma/Beraberlik/Kaybetme olasılıkları)
- Toplam gol sayısı tahmini (2.5 üstü/altı)
- Korner sayısı tahmini
- Karşılıklı gol tahmini
- Skor tahmini

## Teknolojiler

- **Backend:** Flask
- **Makine Öğrenimi:** TensorFlow
- **Veri İşleme:** NumPy, Pandas
- **Frontend:** HTML, CSS, JavaScript
- **API:** Football-Data.org

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/SaltukS1/-prediction.git
cd tahmin
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. `.env` dosyası oluşturun ve Football-Data.org API anahtarınızı ekleyin:
```
FOOTBALL_DATA_API_KEY=your_api_key_here
```

4. Uygulamayı başlatın:
```bash
.\run.bat
```

5. Tarayıcınızda `http://127.0.0.1:5000` adresine gidin.

## Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. 
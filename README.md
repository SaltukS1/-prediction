# Futbol Maç Tahmin Uygulaması

Bu uygulama, futbol maçlarının sonuçlarını tahmin etmek için makine öğrenimi ve istatistiksel modeller kullanan bir web uygulamasıdır.

## Özellikler

- Maç sonuçları tahmini (Kazanma/Beraberlik/Kaybetme olasılıkları)
- Toplam gol sayısı tahmini (1.5, 2.5, 3.5, 4.5 üstü/altı)
- Korner sayısı tahmini (3.5, 4.5, 5.5, 8.5 üstü/altı)
- Karşılıklı gol tahmini (BTTS)
- Skor tahmini
- İlk ve ikinci yarı sonuçları tahmini
- Gol atabileceği tahmin edilen oyuncuların listesi
- Takım karşılaştırması (form, sezon puanları, gol ortalamaları ve son maçlar)
- Canlı maç simülasyonu ve olayları
- Favori takımlar listesi
- Önceki tahminlerin geçmişi
- Karanlık mod

## Teknolojiler

- **Backend:** Flask
- **Makine Öğrenimi:** TensorFlow, Keras
- **Veri İşleme:** NumPy, Pandas, SciKit-Learn
- **Frontend:** HTML, CSS, JavaScript, Bootstrap 5
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

## Yeni Özellikler

### Takım Karşılaştırması
İki takımın istatistiklerini yan yana görerek karşılaştırabilirsiniz. Form puanları, sezon puanları, gol ortalamaları ve son 5 maç performanslarını içerir.

### Favori Takımlar
Sık kullandığınız takımları favori olarak kaydedebilir ve hızlıca seçebilirsiniz. Favoriler tarayıcı hafızasında saklanır.

### Canlı Maç Simülasyonu
Tahmin edilen maçın canlı bir simülasyonunu görebilirsiniz. Simülasyon boyunca:
- Maç olayları (gol, kart, korner, vb.) gerçek zamanlı olarak gösterilir
- Top hakimiyeti grafiği sürekli güncellenir
- Gol atan oyuncular, tahmin modelinden alınan gol atma olasılıklarına göre seçilir

### Karanlık Mod
Uygulama arayüzünü karanlık moda geçirebilirsiniz. Tercihleriniz tarayıcı hafızasında saklanır.

### Poisson Dağılımına Dayalı Tahminler
Gol tahminleri artık Poisson dağılımı kullanılarak yapılmaktadır, bu da daha doğru sonuçlar sağlar.

## Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.

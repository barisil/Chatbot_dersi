📊 TÜİK İstatistik Chatbot

🎥 Tanıtım Videosu  
https://www.youtube.com/watch?v=r3HDQdH1b2w

OpenAI GPT-4o Mini ve RAGAS performans değerlendirmesi ile geliştirilmiş, Türkiye'nin gençlik, aile ve yaşlı istatistiklerine özel RAG tabanlı chatbot.
🎯 Özellikler

Akıllı Döküman Retrieval: Yıl ve kategori bazlı filtreleme ile optimize edilmiş arama
Metadata Tabanlı Sınıflandırma: Her chunk otomatik olarak yıl ve kategori bilgisi ile etiketlenir
Kavram Uyumsuzluk Kontrolü: LLM'in yanlış bilgi üretmesini engelleyen guard mekanizması
Karşılaştırma Desteği: Çoklu yıl sorgularında gelişmiş retrieval stratejisi
RAGAS Entegrasyonu: Chatbot performansını ölçmek için kapsamlı değerlendirme sistemi
Kalıcı Vektör Veritabanı: ChromaDB ile embedding'lerin disk üzerinde saklanması

📁 Proje Yapısı
CHATBOT_DERSI/
│
├── .vscode/                # VS Code yapılandırma dosyaları
├── chroma_db/              # Kalıcı vektör veritabanı (otomatik oluşturulur)
├── data/                   # PDF dokümanlar (TÜİK istatistikleri)
├── pages/                  # Streamlit sayfaları
│   └── ragas_evaluation.py # RAGAS değerlendirme sayfası
│
├── .env                   # Ortam değişkenleri (OPENAI_API_KEY)
├── .gitignore             # Git ignore dosyası
├── pdf_gemini_1.py        # Alternatif implementasyon (Gemini)
├── pdf_gpt_2.py           # Alternatif implementasyon (GPT v2)
├── pdf_gpt_3.py           # ⭐ Ana uygulama dosyası
├── requirements.txt       # Python bağımlılıkları
└── README.md             # Bu dosya
🚀 Kurulum
1. Gereksinimler

Python 3.8+
OpenAI API anahtarı

2. Bağımlılıkların Yüklenmesi
pip install -r requirements.txt

#### Ana Kütüphaneler

- **streamlit**: Web arayüzü için
- **langchain**: RAG pipeline için
- **langchain-openai**: OpenAI entegrasyonu
- **langchain_google_genai** Gemini entegrasyonu (karşılaştırma için)
- **langchain-chroma**: Vektör veritabanı
- **langchain-community**: Döküman yükleyiciler
- **pypdf**: PDF okuma
- **python-dotenv**: Ortam değişkenleri yönetimi
- **ragas**: Performans değerlendirmesi
- **datasets**: RAGAS için veri yönetimi

Tam liste için `requirements.txt` dosyasına bakın.

3. Ortam Değişkenlerinin Ayarlanması
Proje kök dizininde .env dosyası oluşturun:
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. PDF Dokümanların Eklenmesi

`data/` klasörüne TÜİK PDF dokümanlarınızı ekleyin. Dosya isimlendirme formatı:
```
kategori_yıl.pdf
Örnek:

genclik_20.pdf → 2020 Gençlik İstatistikleri
yasli_23.pdf → 2023 Yaşlı İstatistikleri
aile_18.pdf → 2018 Aile İstatistikleri

💻 Kullanım
Ana Uygulamayı Çalıştırma
streamlit run pdf_gpt_3.py
```

Uygulama `http://localhost:8501` adresinde açılacaktır.

### İlk Çalıştırma

1. Uygulama açıldığında PDF'ler otomatik olarak yüklenecek ve vektör veritabanı oluşturulacaktır
2. Bu işlem ilk seferde birkaç dakika sürebilir
3. Sonraki çalıştırmalarda mevcut veritabanı kullanılacağı için hızlı açılır

### Soru Sorma

Chatbot aşağıdaki türde soruları yanıtlayabilir:

**Basit Sorular:**
```
- 2020 yılında genç nüfus oranı nedir?
- 2023 yılında akraba evliliği oranı nedir?
- 2024 yılında yaşlı nüfus kaç kişidir?
```

**Karşılaştırma Soruları:**
```
- 2014 ile 2024 arasında aile yapısı nasıl değişti?
- Yaşlı nüfus oranı yıllara göre nasıl bir trend gösteriyor?
```

**Kategori Bazlı Sorular:**
```
- En son yıl için gençlik istatistikleri nedir?
- Hangi yıllarda evlilik oranı en yüksekti?

📊 RAGAS Değerlendirmesi
Chatbot performansını ölçmek için RAGAS (Retrieval-Augmented Generation Assessment) entegrasyonu mevcuttur.
RAGAS Sayfasına Erişim

Ana chatbot sayfasında en az bir soru sorun
Sağ alt köşedeki "📈 RAGAS Değerlendirmesine Git" butonuna tıklayın
pages/ragas_evaluation.py sayfası açılacaktır

Değerlendirilen Metrikler

Context Precision: Getirilen bağlamın ne kadar ilgili olduğu
Context Recall: Doğru cevap için gerekli bilginin ne kadarının getirildiği
Faithfulness: Cevabın bağlama ne kadar sadık olduğu
Answer Relevancy: Cevabın soruyla ne kadar ilgili olduğu

⚙️ Yapılandırma
pdf_gpt_3.py dosyasındaki sabitler:
CHUNK_SIZE = 600          # Her chunk'ın karakter boyutu
CHUNK_OVERLAP = 120       # Chunk'lar arası örtüşme
TOP_K = 4                 # Döndürülecek maksimum chunk sayısı
Chunk Size ve Overlap Optimizasyonu

CHUNK_SIZE: 500-800 arası değerler genelde iyi sonuç verir
CHUNK_OVERLAP: %20 oranında overlap (CHUNK_SIZE'ın 1/5'i) önerilir
Daha uzun dokümanlar için chunk size artırılabilir
Daha hassas sorgular için overlap artırılabilir

🛡️ Güvenlik Önlemleri
Kavram Uyumsuzluk Kontrolü (guard_mismatch)
LLM'in farklı ama benzer kavramları karıştırmasını önler:

❌ "yaşlı nüfus oranı" ≠ "yaşlı bağımlılık oranı"
❌ "doğuşta beklenen yaşam süresi" ≠ "65 yaşında beklenen yaşam süresi"

Uyumsuzluk tespit edilirse: "Bu bilgi dokümanlarda bulunmamaktadır."
🎨 Özellikler Detayı
1. Akıllı Retrieval (retrieve_docs)
pythondef retrieve_docs(question: str) -> List[Document]:
    # Metinden yıl çıkarma
    # Yıl bazlı filtreleme
    # Karşılaştırma sorularında k değerini artırma
    # Yedek genel arama
    # Tekrar eden sonuçları temizleme
2. Metadata Extraction
Her PDF'den otomatik olarak çıkarılır:

Kategori: genclik, yasli, aile
Yıl: Dosya adından (örn: _20.pdf → 2020)

3. Context Tracking
Her soru-cevap için kaydedilir:

Kullanıcı sorusu
Sistem cevabı
Kullanılan context'ler
Ground truth (varsa)

🔧 Sorun Giderme
Problem: "OPENAI_API_KEY bulunamadı"
Çözüm: .env dosyasını kontrol edin ve API anahtarınızı ekleyin
Problem: "PDF bulunamadı"
Çözüm: data/ klasörüne PDF dosyalarını ekleyin
Problem: Vektör veritabanı yavaş yükleniyor
Çözüm:

İlk yüklemede normaldir
Sonraki çalıştırmalarda chroma_db/ klasörü kullanılır
Yeniden oluşturmak için chroma_db/ klasörünü silin

Problem: Yanlış cevaplar alıyorum
Çözüm:

CHUNK_SIZE ve CHUNK_OVERLAP değerlerini ayarlayın
TOP_K değerini artırın
PDF'lerin doğru formatlandırıldığından emin olun

📝 Ground Truth Test Soruları
Sistem aşağıdaki test soruları için ground truth cevaplarına sahiptir:

2020 yılında genç nüfus oranı nedir?
2023 yılında akraba evliliği oranı nedir?
2014 yılında boşanan çift sayısı kaçtır?
2018 yılında gençlerde işsizlik oranı nedir?
2020 yılında ne eğitimde ne istihdamda olan gençlerin oranı nedir?
2023 yılında internet kullanan gençlerin oranı nedir?
2024 yılında yaşlı nüfus kaç kişidir?

Bu sorular RAGAS değerlendirmesinde kullanılır.
🤝 Katkıda Bulunma

Fork edin
Feature branch oluşturun (git checkout -b feature/amazing-feature)
Commit edin (git commit -m 'Add amazing feature')
Push edin (git push origin feature/amazing-feature)
Pull Request açın

📄 Lisans
Bu proje eğitim amaçlıdır.
📧 İletişim
Sorularınız için issue açabilirsiniz.
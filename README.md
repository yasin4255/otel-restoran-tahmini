#  # 🏞️ Turizm Projesi: Yorumlardan Mekân Türü Eşleştirme

Bu proje, Python dili ve NLTK kütüphanesi kullanılarak doğal dil işleme (NLP) teknikleriyle turizm alanındaki kullanıcı yorumlarını analiz etmeyi ve bu yorumları ilgili mekân türleriyle eşleştirmeyi amaçlamaktadır. Proje kapsamında veri ön işleme, metin temizliği ve temel NLP süreçleri uygulanmıştır.

---

## 1. Hafta — Veri Hazırlama ve Önişleme
1.1. Veri Toplama

ilk olarak otel ve restoran verilerini ayrı ayrı veriler çekilmiştir daha sonra bu verilerin yorum sütunlarına ayrılmıştır.
Ayırılan yorum sütunlarını birleştirerek yeni bir csv (birlesik_yorumlar.csv) dosyasına kaydedilmiştir.
yeni csv dosyası üzerinde aşağıdaki işlemleri uyguladım.
-Bu çalışmam ise "yorumların_birleştirilmesi.ipynb" adlı kaynak dosyasında bulunmaktadır

1.2. Veri Ön İşleme
Yorum verileri üzerinde gerçekleştirilen temel işlemler:

-  Küçük harfe çevirme  
-  Noktalama işaretlerinin kaldırılması  
-  İngilizce stopword (gereksiz kelimeler) temizliği  
-  Tokenizasyon (metni kelimelere ayırma)  
-  Lemmatizasyon ( Kelimeler köklerine indirgenerek farklı çekimler aynı forma getirilmiştir.) 
- Stemming: Kelimenin kökünü bulmak için yapılmıştır

---

## 🔍 Proje Özeti

CSV dosyasından alınan yorumlar şu adımlardan geçirilmiştir:

- Verinin `pandas` ile yüklenmesi ve genel incelemesi
- Eksik verilerin kontrolü
- Cümle ve kelime seviyesinde ayrıştırma (`tokenization`)
- İngilizce stopwords (nltk) ile filtreleme
- Lemmatizasyon ve stemleme işlemleriyle kelimelerin kök formlarının çıkarılması
- Cümle listesi oluşturularak yapısal analiz yapılması
- Veri ön işleme adımları, nltk python, pandas ve re kütüphaneleri kullanılarak Python'da uygulanmıştır.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

- Python 3.10
-  **Python**: 3.12
- **Kütüphaneler**:
  - `nltk==3.9.1`
  - `gensim==4.3.3`
  - `scikit-learn==1.5.1`
  - `pandas==2.2.3`
  - `numpy==1.26.4`
  - `matplotlib==3.9.2`

- Jupyter Notebook
- [NLTK - Natural Language Toolkit](https://www.nltk.org/)
- pandas
- numpy
- import gensim   Word2Vec gibi kelime vektörü modellerini kullanmak için.
- from gensim.models import Word2Vec  # Özellikle Word2Vec modelini kullanmak için
- import pandas as pd  # Veri çerçeveleri (DataFrame) ile çalışmak ve CSV dosyalarını okumak için
- import nltk  # Doğal Dil İşleme (NLP) görevleri için
- from nltk.tokenize import word_tokenize, sent_tokenize  # Metni kelimelere ve cümlelere ayırmak için
- from nltk.corpus import stopwords  # Stop kelimelerini (anlamsız sık kullanılan kelimeler) elde etmek için
- from nltk.stem import WordNetLemmatizer, PorterStemmer  # Kelime köklerini bulmak için (lemmatize ve stem)
- from collections import Counter  # Listelerdeki elemanların sıklığını saymak için


---

##  2. Hafta: TF-IDF Vektörleştirme ve Word2Vec Modelleri 

Bu hafta, ön işlenmiş metin verileri hem TF-IDF yöntemiyle vektörleştirilecek hem de Word2Vec modeli kullanılarak kelime vektörleri elde edilecektir.

# 2.1. TF-IDF Vektörleştirme
TF-IDF (Term Frequency-Inverse Document Frequency) yöntemi, bir metin içindeki kelimelerin önemini ölçmek için kullanılan bir tekniktir. Bu adımda, her bir metin verisi, terim frekansları (TF) ve ters belge frekansı (IDF) kullanılarak bir vektöre dönüştürülür.
sklearn.feature_extraction.text kütüphanesindeki TfidfVectorizer sınıfı, bu dönüşümü gerçekleştirmek için kullanılır.
kod klaörünün içinde bulunan TF-İDF' dosyasında bu işlem gerçekleştirilmiştir. Elde edilen bulgular dosya içinde bulunmaktadır.

## 2.2. Cosine Similarity (Kosinüs Benzerliği) Hesaplaması
.TF-IDF vektörleri elde edildikten sonra, metinler arasındaki benzerliği ölçmek için Cosine 
 Similarity yöntemi kullanılır. Bu yöntem, iki vektör arasındaki açının kosinüsünü hesaplayarak 
 metinlerin ne kadar benzer olduğunu belirler.
.sklearn.metrics.pairwise kütüphanesindeki cosine_similarity fonksiyonu, bu hesaplamayı yapmak 
 için kullanılır. *notebooks klaörünün içinde bulunan 'TF-İDF' dosyasında bu işlem 
 gerçekleştirilmiştir. Elde edilen bulgular dosya içinde bulunmaktadır
 
## 2.3. İlk Cümle için En Yüksek TF-IDF Skorlu Kelimeler
TF-IDF vektörleştirme işleminden sonra, her metindeki en önemli kelimeler belirlenir. Bu, her metin için en yüksek TF-IDF skoruna sahip kelimelerin bulunmasıyla yapılır.
Bu analiz, veri setindeki metinlerin anahtar temalarını ve özelliklerini anlamaya yardımcı olur.

## 2.4. Cosine Similarity Matrisi Oluşturma
Tüm metinler arasındaki Cosine Similarity skorları bir matris içinde düzenlenir. Bu matris, hangi metinlerin birbirine daha çok benzediğini görselleştirmeyi ve analiz etmeyi kolaylaştırır.
Bu matris, öneri sistemleri veya benzer arıza kayıtlarını bulma gibi uygulamalar için temel oluşturabilir.
## 2.5. Word2Vec Modelleri Eğitimi
.Word2Vec modeli, kelimelerin anlamlarını vektörler aracılığıyla temsil etmeyi amaçlayan bir 
 tekniktir. Bu adımda, metin verilerinden kelime vektörleri elde edilir.
.Model eğitimi için farklı parametre kombinasyonları kullanılır. Bu parametreler, modelin 
 performansını ve elde edilen vektörlerin kalitesini etkileyebilir.
.Model eğitimi kod klasörü içerisinde yer alan 'word2vec' dosyasında gerçekleştirilmiştir.
.Seçilecek parametreler şunları içeriyor:
 - Model tipi: CBOW (Continuous Bag of Words) veya Skip-gram.
  -Pencere boyutu: Bir kelimenin bağlamını oluşturan kelime sayısı.
  -Vektör boyutu: Kelimelerin temsil edileceği vektörlerin boyutu.
.Eğitilen modeller, daha sonra kullanılmak üzere dosyaya kaydedilmiştir. Dosya adları, 
 kullanılan parametreleri içerecek şekilde düzenlenmiştir (örneğin, "lemmatized_model_cbow_window2_dim100.model"). Elde edilen dosyalar, model  klasörü içerisine 
 kaydedilmiştir.
# 2.6. Model Değerlendirmesi ve Kullanımı
Eğitilen Word2Vec modelleri, kelime benzerliği, kelime analojisi gibi görevlerde değerlendirilebilir.
Modelin performansı ve elde edilen vektörlerin kalitesi analiz edilebilir.
En iyi performansı gösteren modeller, proje kapsamında kullanılmak üzere seçilebilir.

# Word2Vec Model 

Bu proje, otel ve restoran yorumları üzerinde **Word2Vec** modellerini eğitmek ve kelime benzerliklerini analiz etmek için tasarlanmıştır. Aşağıda projenin temel adımları ve açıklamaları bulunmaktadır.

---

## Adımlar

### 1. Gerekli Kütüphanelerin Kurulumu
- **Kullanılan Araçlar**: `gensim` (Word2Vec modeli için), `pandas` (veri işleme), `nltk` (metin işleme).
- **NLTK Paketleri**: Tokenizasyon, stopwords'ler ve lemmatization için gerekli paketler indirilir.

### 2. Veri Setinin Hazırlanması
- **Veri Kaynakları**: 
  - `lemmatized_sentences.csv`: Kelimelerin kök hallerini içeren cümleler.
  - `stemmed_sentences.csv`: Kelime köklerini içeren cümleler.
- **Temizlik İşlemleri**:
  - NaN ve boş değerler temizlenir.
  - Metinler özel karakterlerden arındırılır, küçük harfe çevrilir.
  - Stopwords'ler ve tek karakterli kelimeler filtrelenir.

### 3. Veri Analizi ve Vektörleştirme
- **Model Parametreleri**:
  - **Model Türü**: CBOW veya Skip-gram.
  - **Pencere Boyutu**: 2 veya 4.
  - **Vektör Boyutu**: 100 veya 300.
- **Eğitim**:
  - Her parametre kombinasyonu için ayrı modeller eğitilir.
  - Modeller `.model` uzantısıyla kaydedilir.
- **Analiz**:
  - "soup" kelimesine en benzer 3 kelime ve skorları çıkarılır.
  - Veri setindeki en sık kullanılan 20 kelime listelenir.

---

## Sonuçlar
- **Kaydedilen Modeller**: `lemmatized_model_cbow_vs100_w2.model`, `stemmed_model_skipgram_vs300_w4.model` gibi isimlerle kaydedilir.
- **Örnek Çıktılar**:
  - Kelime benzerlikleri yüksek skorlarla raporlanır (örneğin, "soup" ↔ "burger": 0.9964).
  - En sık kullanılan kelimeler "good", "staff", "room" gibi tematik terimlerdir.

---

## Nasıl Çalıştırılır?
1. **Veri Yollarını Güncelleyin**: CSV dosyalarının doğru konumunu belirtin.
2. **Jupyter Not Defterini Başlatın**: Tüm hücreleri sırayla çalıştırın.
3. **Sonuçları Görüntüleyin**: Modeller ve analiz çıktıları otomatik olarak oluşturulur.

---
## Görev Tanımları
1. **Giriş Metni Seçimi**: Veri setinden bir yorum seçilir (ör. "The fries were terrific also, hot crisp..."). Bu metin, veri setindeki bir satırdan alınmalıdır.
2. **Benzerlik Hesaplama**:
   - **TF-IDF**: Her metin için TF-IDF vektörleri oluşturulur ve Cosine Similarity ile giriş metnine en benzer 5 metin bulunur.
   - **Word2Vec**: Her metnin kelimelerinin vektörleri modelden alınır, ortalama vektör hesaplanır ve Cosine Similarity ile en benzer 5 metin belirlenir.
3. **Değerlendirme**:
   - **Anlamsal Değerlendirme**: Her modelin önerdiği 5 metne 1-5 arası puan verilir:
     - 1: Çok alakasız
     - 2: Kısmen ilgili
     - 3: Ortalama benzer
     - 4: Anlamlı benzerlik
     - 5: Çok güçlü benzerlik
     - Model başına ortalama puanlar hesaplanır (örn. `[4,4,3,5,4] → Ortalama: 4.0`).
   - **Sıralama Tutarlılığı**: Modellerin en iyi 5 sonucu Jaccard skoru ile karşılaştırılır:
     - Jaccard = Kesişim / Birleşim (örn. 3 ortak metin / 7 toplam metin = 0.43).
     - 18x18 Jaccard matrisi oluşturulur.
4. **Raporlama**: PDF raporunda aşağıdaki başlıklar yer almalıdır:
   - Her modelin en iyi 5 metni ve benzerlik skorları.
   - Anlamsal değerlendirme tablosu ve yorumları.
   - Jaccard matrisi ve analiz.

## Kurulum
Notebook'u çalıştırmak için aşağıdaki adımları izleyin:

1. **Depoyu Klonlama** (varsa):
   ```bash
   git clone <depo-url'si>
   cd <depo-dizini>
   ```

2. **Bağımlılıkları Yükleme**:
   Gerekli Python paketlerini pip ile yükleyin:
   ```bash
   pip install nltk gensim scikit-learn pandas numpy matplotlib
   ```

3. **NLTK Verilerini İndirme**:
   Python ortamında şu komutları çalıştırın:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Verileri Hazırlama**:
   - Aşağıdaki dosyaların doğru dizinlerde olduğundan emin olun:
     - `trip_rest_neywork_1.csv`
     - `hotel_reviews.csv`
     - `lemmatized.csv`
     - `stemmed.csv`
     - Word2Vec model dosyaları (`<model_dir>/*.model`)
   - Gerekirse notebook'taki dosya yollarını güncelleyin (örn. `C:/Users/yasin/Desktop/od/`).

## Kullanım
1. **Notebook'u Açma**:
   ```bash
   jupyter notebook
   ```
   Ardından `metin işleme.ipynb` dosyasını açın.

2. **Hücreleri Çalıştırma**:
   - Hücreleri sırayla çalıştırarak:
     - Bağımlılıkları yükleyin.
     - Veri setlerini birleştirin ve ön işleme yapın.
     - TF-IDF ve Word2Vec modelleriyle benzerlik hesaplayın.
     - Jaccard skoru matrisini oluşturun.
   - Giriş metni olarak veri setinden bir yorum seçin (örn. `input_index` ile).

3. **Çıktılar**:
   - Birleşik veri seti: `birlesik_aciklamalar.csv`
   - Her model için en benzer 5 metin ve Cosine Similarity skorları.
   - 18x18 Jaccard skoru matrisi (modellerin sıralama örtüşmesi).

## Dosya Yapısı
- `yapay_zeka_son.ipynb`: Ana Jupyter notebook.
- `trip_rest_neywork_1.csv`: Restoran yorumları veri seti.
- `hotel_reviews.csv`: Otel yorumları veri seti.
- `birlesik_aciklamalar.csv`: Birleşik veri seti.
- `lemmatized.csv`: Lemmatizasyon uygulanmış veri.
- `stemmed.csv`: Kök bulma uygulanmış veri.
- `<model_dir>/*.model`: Word2Vec modelleri (16 adet: 8 lemmatized, 8 stemmed).

## Notlar
- Dosya yollarının yerel dizin yapınıza uygun olduğundan emin olun.
- Word2Vec modelleri `model_dir` dizininden yüklenir. Modeller yoksa, kodda eğitim adımlarını ekleyin.
- Giriş metni, veri setinden seçilmelidir (`input_index` ile).
- Anlamsal değerlendirme için önerilen metinlere 1-5 arası puan verin ve ortalama skorları hesaplayın.




 

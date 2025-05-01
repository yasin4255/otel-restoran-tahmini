#  # 🏞️ Turizm Projesi: Yorumlardan Mekân Türü Eşleştirme

Bu proje, Python dili ve NLTK kütüphanesi kullanılarak doğal dil işleme (NLP) teknikleriyle turizm alanındaki kullanıcı yorumlarını analiz etmeyi ve bu yorumları ilgili mekân türleriyle eşleştirmeyi amaçlamaktadır. Proje kapsamında veri ön işleme, metin temizliği ve temel NLP süreçleri uygulanmıştır.

---

## 📅 1. Hafta — Veri Hazırlama ve Önişleme
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

- Python 3.x
- Jupyter Notebook
- [NLTK - Natural Language Toolkit](https://www.nltk.org/)
- spaCy (isteğe bağlı)
- pandas
- numpy

---

##  2. Hafta: TF-IDF Vektörleştirme ve Word2Vec Modelleri 

Bu hafta, ön işlenmiş metin verileri hem TF-IDF yöntemiyle vektörleştirilecek hem de Word2Vec modeli kullanılarak kelime vektörleri elde edilecektir.

2.1. TF-IDF Vektörleştirme
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
# 2.5. Word2Vec Modelleri Eğitimi
Word2Vec modeli, kelimelerin anlamlarını vektörler aracılığıyla temsil etmeyi amaçlayan bir tekniktir. Bu adımda, metin verilerinden kelime vektörleri elde edilir.
Model eğitimi için farklı parametre kombinasyonları kullanılır. Bu parametreler, modelin performansını ve elde edilen vektörlerin kalitesini etkileyebilir.
Model eğitimi notebooks klasörü içerisinde yer alan '04_word2vec' dosyasında gerçekleştirilmiştir.
Seçilecek parametreler şunları içeriyor:
Model tipi: CBOW (Continuous Bag of Words) veya Skip-gram.
Pencere boyutu: Bir kelimenin bağlamını oluşturan kelime sayısı.
Vektör boyutu: Kelimelerin temsil edileceği vektörlerin boyutu.
Eğitilen modeller, daha sonra kullanılmak üzere dosyaya kaydedilmiştir. Dosya adları, kullanılan parametreleri içerecek şekilde düzenlenmiştir (örneğin, "lemmatized_model_cbow_window2_dim100.model"). Elde edilen dosyalar, models klasörü içerisine kaydedilmiştir.
# 2.6. Model Değerlendirmesi ve Kullanımı
Eğitilen Word2Vec modelleri, kelime benzerliği, kelime analojisi gibi görevlerde değerlendirilebilir.
Modelin performansı ve elde edilen vektörlerin kalitesi analiz edilebilir.
En iyi performansı gösteren modeller, proje kapsamında kullanılmak üzere seçilebilir.

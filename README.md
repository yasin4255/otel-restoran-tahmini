#  # 🏞️ Turizm Projesi: Yorumlardan Mekân Türü Eşleştirme

Bu proje, Python dili ve NLTK kütüphanesi kullanılarak doğal dil işleme (NLP) teknikleriyle turizm alanındaki kullanıcı yorumlarını analiz etmeyi ve bu yorumları ilgili mekân türleriyle eşleştirmeyi amaçlamaktadır. Proje kapsamında veri ön işleme, metin temizliği ve temel NLP süreçleri uygulanmıştır.

---

## 📅 1. Hafta — Veri Hazırlama ve Temizleme

Yorum verileri üzerinde gerçekleştirilen temel işlemler:

- ✅ Küçük harfe çevirme  
- ✅ Noktalama işaretlerinin kaldırılması  
- ✅ İngilizce stopword (gereksiz kelimeler) temizliği  
- ✅ Tokenizasyon (metni kelimelere ayırma)  
- ✅ Lemmatizasyon ve stemleme (kök forma indirgeme)  
  - Kullanılan kütüphaneler: `nltk`,

---

## 🔍 Proje Özeti

CSV dosyasından alınan yorumlar şu adımlardan geçirilmiştir:

- Verinin `pandas` ile yüklenmesi ve genel incelemesi
- Eksik verilerin kontrolü
- Cümle ve kelime seviyesinde ayrıştırma (`tokenization`)
- İngilizce stopwords (nltk) ile filtreleme
- Lemmatizasyon ve stemleme işlemleriyle kelimelerin kök formlarının çıkarılması
- Cümle listesi oluşturularak yapısal analiz yapılması

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

- Python 3.x
- Jupyter Notebook
- [NLTK - Natural Language Toolkit](https://www.nltk.org/)
- spaCy (isteğe bağlı)
- pandas
- numpy

---

## 📁 Veri Seti

Projede kullanılan veri seti: **`birlesik_yorumlar.csv`**

**Not:** Dosya, projeyle aynı dizinde olmalıdır. Eğer çalıştırırken yol hatası alırsanız, aşağıdaki gibi düzenleyin:

```python
df = pd.read_csv("birlesik_yorumlar.csv")


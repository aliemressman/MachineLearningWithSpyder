# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:32:35 2025

@author: aliem
"""

import pandas as pd
import numpy as np

# VERİ ÖN İŞLEME
import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer() # Kelimelerdeki gereksiz ekleri atacak ve köküne ulaşacak.

nltk.download("stopwords")
from nltk.corpus import stopwords # Gereksiz kelimeleri cümleden çıkartacak.

yorumlar = pd.read_csv("resteurant_Reviews.csv", on_bad_lines='skip')

derlem = []

for i in range(716):
    yorum = re.sub("[^a-zA-Z]"," ", yorumlar["Review"][i]) # Noktalama işaretlerinden kurtulduk.
    yorum = yorum.lower() # Her harfi küçük harfe çevirdik.
    yorum = yorum.split() # Listeye çevirdik.
    # Önce for döngüsü ile yorumda bulunan her kelime döndürülür. Sonra if ile kelimenin stopword olup olmadığı
    # kontrol edilir. Eğer kelime stopwords değilse ps.stem(kelime) komutu çalışır ve kelimem köküne çevrilir.
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum) # Split ile ayrılan kelimeler birleştiriliyor.
    derlem.append(yorum)


# ÖZNİTELİK ÇIKARIMI(Feature Extraction)
# Bag of Words(BOW - Kelime Çantası)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000) # Max 2000 farklı kelime
X = cv.fit_transform(derlem).toarray() # Bağımsız Değişken

# 1. sütunun (col 1) ortalamasını hesapla (NaN olmayan değerler üzerinden)
mean_value = yorumlar.iloc[:, 1].mean()
# NaN değerleri ortalama değer ile değiştir
yorumlar.iloc[:, 1].fillna(mean_value, inplace=True)
y = yorumlar.iloc[:,1].values.astype(int) # Bağımlı Değişken

# MAKİNE ÖĞRENMESİ
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_predict = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
print(cm) #◘ %70 doğru












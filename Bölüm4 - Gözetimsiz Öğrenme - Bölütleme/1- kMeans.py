# -*- coding: utf-8 -*-
"""
K-Means Kümeleme Algoritması ile Müşteri Segmentasyonu

Bu kod, "musteriler.csv" dosyasındaki verileri kullanarak müşterileri segmentlere ayırır
ve Elbow Method'u kullanarak en uygun küme sayısını belirler.

Oluşturan: aliem
Tarih: 31 Ocak 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Veri setini oku
veriler = pd.read_csv("musteriler.csv")

# Kullanılacak özellikleri seç (3. sütundan sonraki tüm sütunlar)
X = veriler.iloc[:, 3:].values

# K-Means modelini oluştur ve eğit
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
kmeans.fit(X)

# Küme merkezlerini yazdır
print("Küme Merkezleri:")
print(kmeans.cluster_centers_)

# Elbow Method ile en uygun küme sayısını belirleme
sonuclar = []
kume_sayilari = range(1, 11)

for i in kume_sayilari:
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)  # WCSS değerini kaydet

# Elbow Method grafiğini çizdir
plt.figure(figsize=(8,5))
plt.plot(kume_sayilari, sonuclar, marker='o', linestyle='-', color='b')
plt.xlabel("Küme Sayısı")
plt.ylabel("WCSS (İçsel Küme Karesel Toplamı)")
plt.title("Elbow Yöntemi ile En İyi Küme Sayısının Belirlenmesi")
plt.grid()
plt.show()

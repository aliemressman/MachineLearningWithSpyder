# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:59:06 2024

@author: aliem
"""

# Ders 6 Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri Yükleme
veriler = pd.read_csv('eksikVeriler.csv.csv')
print(veriler)

# Eksik Veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)

# fit ve transform metodlarını doğru şekilde kullanma
imputer = imputer.fit(Yas[:,1:4])  # fit metodu çağırılıyor
Yas[:,1:4] = imputer.transform(Yas[:,1:4])  # transform metodu çağırılıyor
print(Yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
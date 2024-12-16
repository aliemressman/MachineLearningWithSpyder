# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:59:06 2024

@author: aliem
"""

# 1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Önişleme
#2.1 Veri Yükleme

veriler = pd.read_csv('eksikveriler.csv.csv')
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

#eksik veriler
#sci - kit learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

# Encoder: Kategorik Veriler -> Numeric(Sayısal) Veriler
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# Numpy dizileri DataFrame Dönüşümü
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

# DataFrame Birleştirme İşlemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

# Verileri eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
# X bağımsız değişken, y bağımlı değişken
# X değerlerim -> ulke,yas,boy,kilo
# Y değerlerim -> Cinsiyet 
# X değerlerim ile Y değerimi bulmaya çalışıyorum.
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state=0)

# Ölçeklendirme işlemi -> Dataların değerlerini birbirine yaklaştırıyoruz.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:35:13 2025

@author: aliem
"""

import pandas as pd

url = "https://bilkav.com/satislar.csv"

datalar = pd.read_csv(url)

X = datalar.iloc[:,0:1].values
Y = datalar.iloc[:,1].values

bolme = 0.33

from sklearn import model_selection
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=bolme, random_state=0)
"""
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()

lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)
print(y_pred)
"""

 # MODELİ KAYDEDİYORUZ. VE KAPATIP AÇTIĞIMIZDA TEKRARDAN MODELİ FİT İLE EĞİTMEMİZE GEREK KALMIYOR.
import pickle
"""
dosya = "model.kayit"
pickle.dump(lr,open(dosya,"wb"))
"""
yuklenen = pickle.load(open("model.kayit","rb"))
print(yuklenen.predict(X_test))
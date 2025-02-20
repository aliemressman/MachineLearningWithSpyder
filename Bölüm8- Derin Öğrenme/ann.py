# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:52:45 2025

@author: aliem
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi1
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Yapay Sinir Ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

classfier = Sequential()

classfier.add(Dense(6, kernel_initializer="uniform", activation="relu", input_dim=11))
classfier.add(Dense(6, kernel_initializer="uniform", activation="relu"))
classfier.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

classfier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

classfier.fit(X_train, y_train, epochs=50, verbose=1)

y_pred = classfier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



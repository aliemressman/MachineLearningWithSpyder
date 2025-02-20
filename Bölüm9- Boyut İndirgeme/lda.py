# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:47:07 2025

@author: aliem
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Wine.csv')
print(veriler)

#veri on isleme
X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components= 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# PCA dönüşümünden önce gelen LogisticReg
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# PCA dönüşümünden sonra gelen LogisticReg
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

# Tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

# Actual / PCA Olmadan Çıkan sonuç
print("Gerçek / PCA'sız\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Actual / PCA ile Çıkan sonuç
print("Gerçek / PCA'lı\n")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

# Actual / PCA ile Çıkan sonuç
print("PCA'sız / PCA'lı\n")
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# LDA Dönüşümünden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

# LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

# LDA Sonrası / orjinal
print("LDA ve sonrası\n")
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)





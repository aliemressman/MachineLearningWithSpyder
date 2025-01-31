# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:59:06 2024

@author: aliem
"""

# 1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv.csv')

x = veriler.iloc[:,1:4].values # bagımsız degislkenler
y = veriler.iloc[:, 4:].values # bagımlı degiskenler
print(y)

# Verileri eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=0)

#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Logistic Regresyon Algoritması
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train) # Eğitim

y_pred = logr.predict(X_test) # Tahmin
print(y_pred)
print(y_test)

# Confusion Matrix (Karmaşıklık Matrisi)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Knn algoritması
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 1, metric = "minkowski")
knn.fit(x_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

# SVC Algoritmasi(SVM in Classifier Versiyonu)

from sklearn.svm import SVC
svc = SVC()
svc = SVC(kernel = "rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)

# Naive Bayes Algoritması
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("GNB")
print(cm)

# DecisionTree Algoritması
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion= "entropy")
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)

# RandomForest Algoritması
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("RFC")
print(cm)

# 7. ROC , TPR, FPR değerleri 
y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)


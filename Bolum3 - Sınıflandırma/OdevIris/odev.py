# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:17:09 2025

@author: aliem
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

veriler = pd.read_excel("Iris.xls")

x = veriler.iloc[:,:4].values;
y = veriler.iloc[:,-1:].values;

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state= 0);

# SCALE EDİLMİŞ DATALARIM
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Logistic Regresyon Algoritması
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train) # Eğitim

y_pred = logr.predict(X_test) # Tahmin

# Confusion Matrix (Karmaşıklık Matrisi)
cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Loogistic:", accuracy)

#--------------------------------------------------------------------------
# Knn algoritması
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 1, metric = "minkowski")
knn.fit(x_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy KNN:", accuracy)

#--------------------------------------------------------------------------

# SVC Algoritmasi(SVM in Classifier Versiyonu)
svc = SVC()
svc = SVC(kernel = "rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy SVC:", accuracy)

#--------------------------------------------------------------------------

# Naive Bayes Algoritması
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy Bayes:", accuracy)

#--------------------------------------------------------------------------

# DecisionTree Algoritması
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion= "entropy")
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy DecisionTree:", accuracy)
#--------------------------------------------------------------------------

# RandomForest Algoritması
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5,criterion="log_loss")
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy RandomForest:", accuracy)









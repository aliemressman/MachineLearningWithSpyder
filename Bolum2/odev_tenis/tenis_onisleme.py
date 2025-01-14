
# 1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Önişleme
#2.1 Veri Yükleme

veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
#veri on isleme
#eksik veriler
#sci - kit learn
print(veriler)

# Encoder: Kategorik Veriler -> Numeric(Sayısal) Veriler
from sklearn import preprocessing

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform) #verilerin hepsini labelEncode ettik

c = veriler2.iloc[:,:1];

from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c) #Hava durumlarını OneHotEncode ettik. Yani hepsini ayırdık.

havaDurumu = pd.DataFrame(data = c, index = range(14),columns= ["o","r","s"]);
sonVeriler = pd.concat([havaDurumu, veriler.iloc[:,1:3]],axis= 1);
sonVeriler = pd.concat([veriler2.iloc[:,-2:],sonVeriler],axis = 1);


# Verileri eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
# X bağımsız değişken, y bağımlı değişken
# X değerlerim -> ulke,yas,boy,kilo
# Y değerlerim -> Cinsiyet 
# X değerlerim ile Y değerimi bulmaya çalışıyorum.
x_train,x_test,y_train,y_test = train_test_split(sonVeriler.iloc[:,:-1],sonVeriler.iloc[:,-1:],test_size = 0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train) # x_train datamdan y_train datamı ÖĞREN.

y_pred = regressor.predict(x_test) # Yukarıda öğrendiğin yöntem ile x_test üzerinden y_test değerlerini tahmin et.
# Yani burada y_pred ile y_test değerlerini karşılaştırıcaz. Kontrol ettiğimiz zaman 8 Tahmin üzerinden 2 yanlış var. 
print(y_pred)

# Backward Elemation
import statsmodels.api as sm 

X = np.append(arr = np.ones((14,1)).astype(int),values = sonVeriler.iloc[:,:-1],axis = 1) # y = b0 + b1x 'deki sabit b0 değerini oluşturduk.

x_l = sonVeriler.iloc[:,[0,1,2,3,4,5]].values
x_l = np.array(x_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],x_l).fit()
print(model.summary())

sonVeriler = sonVeriler.iloc[:,1:]

import statsmodels.api as sm 

X = np.append(arr = np.ones((14,1)).astype(int),values = sonVeriler.iloc[:,:-1],axis = 1) # y = b0 + b1x 'deki sabit b0 değerini oluşturduk.

x_l = sonVeriler.iloc[:,[0,1,2,3,4]].values
x_l = np.array(x_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],x_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
y_test = y_test.iloc[:,1:]

regressor.fit(x_train, y_train) # x_train datamdan y_train datamı ÖĞREN.

y_pred = regressor.predict(x_test)









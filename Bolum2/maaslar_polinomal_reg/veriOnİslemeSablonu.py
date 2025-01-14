# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:18:08 2025

@author: aliem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri Yükleme
veriler = pd.read_csv("maaslar.csv")

# Dataframe dilimleme (slicing)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,-1:]

# Numpy array dönüşümü
X = x.values
Y = y.values

# Lineer Regression
# Doğrusal model oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# Polinomal Regresyon
# Doğrusal olmayan model oluşturma
# 2.Dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg2= PolynomialFeatures(degree= 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)

# 4. Dereceden polinom 
poly_reg3 = PolynomialFeatures(degree= 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


plt.scatter(X,Y, color = "red")
plt.plot(x,lin_reg.predict(X),color = "green")
plt.show()

plt.scatter(X,Y,color = "red")
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)),color = "blue")
plt.show()

plt.scatter(X,Y,color = "red")
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color = "blue")
plt.show()

# Tahminler 

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg3.predict(poly_reg3.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg3.fit_transform([[11]])))





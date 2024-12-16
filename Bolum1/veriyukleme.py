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

veriler = pd.read_csv('veriler.csv.csv')
print(veriler);

# Veri Önişleme
boy = veriler[['boy']]
print(boy)

kiloboy = veriler[['boy','kilo']]
print(kiloboy)
x = 10
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:34:38 2025

@author: aliem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("adsCTROptimisation.csv")

import random

N = 10000
d = 10
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # Verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
    
plt.hist(secilenler)
plt.show()

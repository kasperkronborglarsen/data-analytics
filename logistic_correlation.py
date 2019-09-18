# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
4.6.1

Kasper Kronborg Larsen
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Kasper/Desktop/datasets/Smarket.csv", index_col=0, parse_dates=True)
summary = df.describe()
print(summary.transpose())

corr = df.corr()
print(corr)

plt.figure(1)
plt.plot(df.Volume, 'o', markersize=2)
plt.title('Change in stock trades over time')
plt.xlabel('Days')
plt.ylabel('Volume (in billions)')
plt.show()

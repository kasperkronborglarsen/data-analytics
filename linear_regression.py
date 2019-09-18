# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
3.6.2

Kasper Kronborg Larsen
"""
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import pandas as pd

readdata = csv.reader(open("C:/Users/Kasper/Desktop/datasets/Boston.csv"))

data=[]

for row in readdata:
    data.append(row)
    
Header = data[0]
data.pop(0)

print (pd.DataFrame(data, columns=Header))

medv = []
lstat = []

for i in range(len(data)):
  medv.append(float(data[i][14]))
  lstat.append(float(data[i][13]))

import statsmodels.api as sm

X = lstat
y = medv
X = sm.add_constant(X)
X_1=X[0]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

model.summary()

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

plt.scatter(lstat, medv, s=3)
plt.xlabel("medv")
plt.ylabel("lstat")
plt.plot(lstat, predictions, color='black')

residuals = y - predictions
plt.scatter(predictions, residuals, s=2)
plt.hlines(y=0, xmin=-2, xmax=34, color="black")
plt.xlabel("Predicted")
plt.ylabel("Residuals")


max_value = max(residuals) #Finds largest residual and index 
max_index = residuals.tolist().index(max_value)

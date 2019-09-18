# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
5.3.3

Kasper Kronborg Larsen
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

# Read data
data = pd.read_csv("C:/Users/Kasper/Desktop/datasets/Auto.csv", na_values='?').dropna()

# Changing the column to start from 0
X = data['horsepower'].values.reshape(-1,1)
y = data['mpg'].values.reshape(-1,1)

# Linear regression
lm = skl_lm.LinearRegression()

# Choosing k=10 in k-fold cross validation
crossval = KFold(n_splits=10, random_state=1, shuffle=False)

# Fitting model with polynomial terms of degrees 1-10 and printing MSE and STD for each case
for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=crossval, n_jobs=1)
    print("Degree "+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))
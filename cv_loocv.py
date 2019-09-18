# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
5.3.2

Kasper Kronborg Larsen
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv(open("C:/Users/Kasper/Desktop/datasets/Auto.csv"), na_values='?').dropna()

loo = LeaveOneOut()
# Changing the column to start from 0
X = data['horsepower'].values.reshape(-1,1)
y = data['mpg'].values.reshape(-1,1)

lm = skl_lm.LinearRegression()
model = lm.fit(X,y)

loo.get_n_splits(X)

crossval = KFold(n_splits=392, random_state=None, shuffle=False)

scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=crossval, n_jobs=1)

print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))

for i in range(1,6):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=crossval, n_jobs=1)
    
    print("Degree-" + str(i) + " polynomial MSE: " + str(np.mean(scores)) + ", STD: " + str(np.std(scores)))

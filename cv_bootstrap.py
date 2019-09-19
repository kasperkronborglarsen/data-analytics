# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
5.3.4

Kasper Kronborg Larsen
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import statsmodels.api as sm

# Estimating the accuracy of a linear regression model
# Read data
data = pd.read_csv('C:/Users/Kasper/Desktop/datasets/Auto.csv')

x = data['horsepower'].values.reshape(-1, 1)
y = data['mpg'].values.reshape(-1, 1)
    
# create linear model and summary
x = sm.add_constant(x, prepend=False)
model = sm.OLS(y,x)
result = model.fit()
result.summary()

# Bootstrap function used for linear regression model using 1000 repetetions 
def btstrap_lin_reg(x,y):
    N_rep = 1000
    intercepts = []
    coefs = []
    for i in range(0,N_rep):
        #Pick 1000 samples from x and y
        xSample, ySample = resample(x, y, n_samples=1000)
        regr = LinearRegression()
        clf = regr.fit(xSample,ySample)
        intercepts.append(clf.intercept_[0]) 
        coefs.append(clf.coef_[0][0])
    
    return [np.mean(intercepts), np.mean(coefs), stats.sem(intercepts), stats.sem(coefs)]

# Using bootstrap to estimate intercept and slope
result =  btstrap_lin_reg(x,y)
print('Intercept - bootstrap:', result[0], ' std. err: ', result[2])
print('Coefficients - bootstrap:', result[1], ' std. err: ', result[3])

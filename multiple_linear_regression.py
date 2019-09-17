# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
3.6.2

Kasper Kronborg Larsen
"""
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv(open("C:/Users/Kasper/Desktop/datasets/boston/Boston.csv"))

    
y = data['medv']
X = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']]

## Add an intercept to our model
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

# Print out the statistics
model.summary()



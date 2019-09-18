# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
4.6.3

Kasper Kronborg Larsen
"""

import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report

#Import Smarket data: 
df = pd.read_csv("C:/Users/Kasper/Desktop/datasets/Smarket.csv", usecols=range(1,10), index_col=0, parse_dates=True)
df.head()

#Splits data into training set and test set. 
#Data from 2004 and before are used as training set, and data from 2005 are used as test set. 
X_train = df[:'2004'][['Lag1','Lag2']]
y_train = df[:'2004']['Direction']

X_test = df['2005':][['Lag1','Lag2']]
y_test = df['2005':]['Direction']

#Performs linear discriminant analysis on the training set 
lda = LinearDiscriminantAnalysis()
model = lda.fit(X_train, y_train)

#Prior probabilities of groups: 
print(model.priors_)

#Group means: 
print(model.means_)

#Coefficients of linear discriminants:
print(model.coef_)

#Performs predictions on the test set: 
pred=model.predict(X_test)
print(np.unique(pred, return_counts=True))

#Confusion matrix for predictions: 
print(confusion_matrix(pred, y_test))

#Classification details: 
print(classification_report(y_test, pred, digits=3))

#Predicted probabilities: 
pred_p = model.predict_proba(X_test)

#Applying 50% threshold to posterior probabilities allows us 
#to recreate predictions: 
print(np.unique(pred_p[:,1]>0.5, return_counts=True))

#Applying 90% threshold: 
print(np.unique(pred_p[:,1]>0.9, return_counts=True))

#Findind maximum posterior probability: 
max(pred_p[:,1])

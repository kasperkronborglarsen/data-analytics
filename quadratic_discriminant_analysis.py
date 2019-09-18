# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
4.6.4

Kasper Kronborg Larsen
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data = pd.read_csv("C:/Users/Kasper/Desktop/datasets/Smarket.csv", usecols = range(1,10), parse_dates=True)

#Training data before 2005, test data in 2005
x_train = data[data.Year < 2005][['Lag1', 'Lag2']]
y_train = data[data.Year < 2005]['Direction']
x_test = data[data.Year==2005][['Lag1', 'Lag2']]
y_test = data[data.Year==2005]['Direction']

qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)

# Prior probabilities of groups
print("Down : %f" % qda.priors_[0])
print("Up : %f" % qda.priors_[1])

# Group means
means = pd.DataFrame(qda.means_,['Down', 'Up'], ['Lag1', 'Lag2'])
print(means)

# Predict test data
y_predict = qda.predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, y_predict).T,['Down', 'Up'], ['Down', 'Up']))
print(classification_report(y_test, y_predict, digits=3))

# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
4.6.5

Kasper Kronborg Larsen
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("C:/Users/Kasper/Desktop/datasets/Smarket.csv", usecols = range(1,10))

#Training data before 2005, test data in 2005
x_train = data[data.Year < 2005][['Lag1', 'Lag2']]
y_train = data[data.Year < 2005]['Direction']
x_test = data[data.Year==2005][['Lag1', 'Lag2']]
y_test = data[data.Year==2005]['Direction']

# n = 1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train,y_train)

y_predict1 = knn1.predict(x_test)
print("Classification Report for k=1")
print(classification_report(y_test, y_predict1, digits=3))
print(pd.DataFrame(confusion_matrix(y_test, y_predict1).T,['Down', 'Up'], ['Down', 'Up']))

# n = 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train,y_train)

y_predict3 = knn3.predict(x_test)
print("Classification Report for k=3")
print(classification_report(y_test, y_predict3, digits=3))
print(pd.DataFrame(confusion_matrix(y_test, y_predict3).T,['Down', 'Up'], ['Down', 'Up']))

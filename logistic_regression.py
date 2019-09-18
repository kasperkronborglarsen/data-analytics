# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
4.6.2

Kasper Kronborg Larsen
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

data = pd.read_csv("C:/Users/Kasper/Desktop/datasets/Smarket.csv", usecols = range(1,10))

# To get the values to fit the answer
for x in range(0, data.Direction.size):
  if data.Direction[x].lower() in ['up']: 
    data.loc[x,'Direction'] = 1
  else:
    data.loc[x,'Direction'] = 0
    
glm = smf.glm("Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume",data,family=sm.families.Binomial()).fit()
print(glm.summary())

pred = glm.predict()
pred_bools = [1 if x > 0.5 else 0 for x in pred]

true_values = data.Direction
confusion_mat = confusion_matrix(true_values, pred_bools)
print(np.transpose(confusion_mat))

prob = (confusion_mat[1,1]+confusion_mat[0,0])/sum(sum(confusion_mat))
print(prob)

# Train with data before 2005
train_data = data[0:sum(data.Year<2005)]

 # test_data is used in the confusion matrix as train_data is unbalanced
test_data = data[sum(data.Year<2005):]

glm_train = smf.glm("Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume",data=train_data,family=sm.families.Binomial()).fit()
print(glm_train.summary())
print(glm_train.params)

pred_train = glm_train.predict(test_data)
pred_bools_train = [1 if x > 0.5 else 0 for x in pred_train]

true_values = test_data.Direction
confusion_mat_train = confusion_matrix(pred_bools_train,true_values)
print(confusion_mat_train)

prob_train = (confusion_mat_train[1,1]+confusion_mat_train[0,0])/sum(sum(confusion_mat_train))
print(prob_train)

# Using Lag1 and Lag2, as they have the highest prediction power
glm_train2 = smf.glm("Direction ~ Lag1 + Lag2",data=train_data,family=sm.families.Binomial()).fit()
print(glm_train2.summary())
print(glm_train2.params)

pred_train2 = glm_train2.predict(test_data)
pred_bools_train2 = [1 if x > 0.5 else 0 for x in pred_train2]

true_values = test_data.Direction
confusion_mat_train2 = confusion_matrix(pred_bools_train2,true_values)
print(confusion_mat_train2)

prob_train2 = (confusion_mat_train2[1,1]+confusion_mat_train2[0,0])/sum(sum(confusion_mat_train2))
print(prob_train2)

print(glm_train2.predict(pd.DataFrame([[1.2, 1.1], [1.5, -0.8]], columns = ["Lag1", "Lag2"])))

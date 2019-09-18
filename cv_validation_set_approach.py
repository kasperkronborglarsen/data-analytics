# -*- coding: utf-8 -*-
"""
Decision Sypport Systems 
5.3.1

Kasper Kronborg Larsen
"""
import pandas as pd
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv(open("C:/Users/Kasper/Desktop/datasets/Auto.csv"), na_values='?').dropna()
data.info()

# We split the set of observations into two halves
train_data = data.sample(196, random_state=1)
test_data = data[~data.isin(train_data)].dropna(how = 'all')

# Changing the column to start from 0
X_train = train_data['horsepower'].values.reshape(-1,1)
y_train = train_data['mpg'].values.reshape(-1,1)
X_test = test_data['horsepower'].values.reshape(-1,1)
y_test = test_data['mpg'].values.reshape(-1,1)

# Using linear regression on the training data only
lm = skl_lm.LinearRegression()
model = lm.fit(X_train, y_train)

# Estimating the response for the test observations
pred = model.predict(X_test)

MSE_lin = mean_squared_error(y_test, pred)

print("Linear MSE: ", MSE_lin)

# Quadratic
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

model = lm.fit(X_train_poly, y_train)

MSE_quad = mean_squared_error(y_test, model.predict(X_test_poly))

print("Quadratic MSE: ", MSE_quad)

# Cubic
poly = PolynomialFeatures(degree=3)
X_train_poly2 = poly.fit_transform(X_train)
X_test_poly2 = poly.fit_transform(X_test)

model = lm.fit(X_train_poly2, y_train)

MSE_cub = mean_squared_error(y_test, model.predict(X_test_poly2))
print("Cubic MSE: ", MSE_cub)

# Try choosing a different random dataset
# train_data = data.sample(196, random_state=2)

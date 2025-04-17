#lab2

import pandas as pd

train = pd.read_csv("lab2/preprocessed.csv")
test = pd.read_csv("lab2/preprocessed_test.csv")

x_train = train.drop(['Age'], axis='columns')
y_train = train['Age']

x_test = test.drop(['Age'], axis='columns')
y_test = test['Age']


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(x_train, y_train)

y_pred_test = linear_model.predict(x_test)
y_pred_train = linear_model.predict(x_train)

import sklearn.metrics as metrics

MSE = metrics.mean_squared_error(y_test, y_pred_test)
print("MSE test", MSE)
MSEt = metrics.mean_squared_error(y_train, y_pred_train)
print("MSE train", MSEt)

import math
print("RMSE test", metrics.root_mean_squared_error(y_test, y_pred_test))
print("RMSE train", math.sqrt(MSEt))

MAE = metrics.mean_absolute_error(y_test, y_pred_test)
print("MAE test", MAE)
MAEt = metrics.mean_absolute_error(y_train, y_pred_train)
print("MAE train", MAEt)
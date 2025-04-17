
import pandas as pd

train = pd.read_csv("lab2/preprocessed.csv")
test = pd.read_csv("lab2/preprocessed_test.csv")

x_train = train.drop(['Transported_int'], axis='columns')
y_train = train['Transported_int']

x_test = test.drop(['Transported_int'], axis='columns')
y_test = test['Transported_int']

from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression()

logreg_model.fit(x_train, y_train)

y_pred_train = logreg_model.predict(x_train)
y_pred_test = logreg_model.predict(x_test)

from sklearn.metrics import classification_report

print("test report")
report = classification_report(y_test, y_pred_test)
print(report)

print("train report")
reportTrain = classification_report(y_train, y_pred_train)
print(reportTrain)
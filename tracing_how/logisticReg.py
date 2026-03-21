from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


# from sklearn.model_selection import train_test_split
# from sklearn import datasets, linear_model, metrics
#
# digits = datasets.load_digits()
#
# X = digits.data
# y = digits.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
#
# reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
# reg.fit(X_train, y_train)
#
# y_pred = reg.predict(X_test)
#
# print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
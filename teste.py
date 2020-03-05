from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
print(X,y)
clf = LogisticRegression()
clf.fit(X, y)
print(clf.predict(X[:2, :]))
print(clf.predict_proba(X[:2, :]))
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X,y = load_iris(return_X_y=True)
clf = LogisticRegression()
clf.fit(X, y)

print(X[0:2, :])
print(clf.predict(X[0:2, :]))
print(y[0:2])

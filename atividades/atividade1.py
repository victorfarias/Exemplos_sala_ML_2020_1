import numpy as np

def RSS(y_true, y_pred):
    e = y_true - y_pred # resíduos
    return np.sum(e*e)

def MAE(y_true, y_pred):
    e = y_true - y_pred # resíduos
    return np.mean(np.absolute(e))

class SimpleLinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        y_bar = np.mean(y)
        x_bar = np.mean(X)

        x_x_bar = X - x_bar
        y_y_bar = y - y_bar
        
        num = np.sum(x_x_bar * y_y_bar)

        denom = np.sum(x_x_bar * x_x_bar)

        self.b1 = num/denom
        self.b0 = y_bar - self.b1 * x_bar

    def predict(self, X):
        return self.b0 + X*self.b1

class SimpleLinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        y_bar = np.mean(y)
        x_bar = np.mean(X)

        x_x_bar = X - x_bar
        y_y_bar = y - y_bar
        
        num = np.sum(x_x_bar * y_y_bar)

        denom = np.sum(x_x_bar * x_x_bar)

        self.b1 = num/denom
        self.b0 = y_bar - self.b1 * x_bar

    def predict(self, X):
        return self.b0 + X*self.b1

class LinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]

        self.b = np.linalg.pinv(X_.T @ X_) @ X_.T @ y

    def predict(self, X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]

        return X_ @ self.b


data = np.loadtxt("./data/advertising.csv", skiprows=1, delimiter=",")
# Regressão linear simples
X_tv = data[:, 1]
y = data[:, -1]

reg = SimpleLinearRegression()
reg.fit(X_tv, y)
y_pred = reg.predict(X_tv)

print(RSS(y, y_pred))
print(MAE(y, y_pred))


# Regressão linear múltipla
X = data[:, 1:4]
y = data[:, -1]
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
print(y_pred)

print(RSS(y, y_pred))
print(MAE(y, y_pred))


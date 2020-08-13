import numpy as np

def MSE(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

class LinearRegressionGD():

    def __init__(self, alpha=0.001, epochs=30000):
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X, y):
        self.w0 = 0
        self.w1 = 0
        self.MSEs = []
        n = X.shape[0]

        for i in range(self.epochs):            
            y_pred = self.w0 + self.w1*X
            e = y - y_pred
            # print(e)

            self.MSEs.append(MSE(y, y_pred))
            
            g0 = (self.alpha/n)*np.sum(e)
            g1 = (self.alpha/n)*np.sum(e*X) # multiplicao elemento a elemento (element wise)
            
            self.w0 = self.w0 + g0
            self.w1 = self.w1 + g1    

    def predict(self, X):
        return self.w0 + self.w1*X


def test_reg_lin_uni():
    data = np.loadtxt("./housing.data")
    data_ = data[:, -2:]
    np.random.shuffle(data_)
    n = data_.shape[0]
    data_train = data_[:int(0.8*n), :]
    data_test = data_[int(0.8*n):, :]
    
    X_train = data_train[:, 0]
    y_train = data_train[:,1]

    X_test = data_test[:, 0]
    y_test = data_test[:,1]

    reg = LinearRegressionGD()
    reg.fit(X_train,y_train)  
    y_test_pred = reg.predict(X_test) 
    print(MSE(y_test, y_test_pred))
    print(reg.MSEs[-1])
    print(reg.w0, reg.w1)
    print(reg.MSEs)


class LinearRegressionGDMV():

    def __init__(self, alpha= 0.01, epochs=1000):
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]

        X_ = np.c_[np.ones(n), X]
        self.w = np.zeros(m+1)

        self.MSEs = []
        
        for i in range(self.epochs):            
            y_pred = X_ @ self.w  # multiplicao de matriz     
            self.MSEs.append(MSE(y, y_pred))
            e = (y-y_pred)
            g = X_.T @ e                        
            self.w = self.w + (self.alpha/n)*g        


def test_reg_lin_multi():
    data = np.loadtxt("./trab1_data.txt")
    np.random.shuffle(data)

    n = data.shape[0]
    data_train = data[:int(0.8*n), :]
    data_test = data[int(0.8*n):, :]

    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    
    X_test = data_test[:, :-1]
    y_test = data_train[:, -1]

    reg = LinearRegressionGDMV()
    reg.fit(X_train, y_train)
    print(reg.MSEs)


class LinearRegressionGDSMV():

    def __init__(self, epochs=2000, alpha=0.01):
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]

        X_ = np.c_[np.ones(n), X]
        self.w = np.zeros(m+1)

        self.MSEs = []
        
        for _ in range(self.epochs):

            y_pred = X_ @ self.w  # multiplicao de matriz     
            self.MSEs.append(MSE(y, y_pred))

            for i in range(n):
                xi = X_[i, :]
                yi_pred = self.w.T @ xi  #  multiplicao matriz
                ei = y[i]-yi_pred
                gi = self.alpha*ei*xi
                self.w = self.w + gi    

            indices = np.arange(n)
            np.random.shuffle(indices)
            X_ = X_[indices, :]
            y = y[indices]

def test_reg_lin_multi_st():
    data = np.loadtxt("./trab1_data.txt")
    np.random.shuffle(data)

    n = data.shape[0]
    data_train = data[:int(0.8*n), :]
    data_test = data[int(0.8*n):, :]

    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    
    X_test = data_test[:, :-1]
    y_test = data_train[:, -1]
    reg = LinearRegressionGDSMV()
    reg.fit(X_train, y_train)
    print(reg.MSEs)

if __name__ == "__main__":
    test_reg_lin_uni()
    test_reg_lin_multi()    
    test_reg_lin_multi_st()

import numpy as np

def acc(y_real, y_pred):
    n = y_real.shape[0]
    n_true = np.count_nonzero(y_real == y_pred)
    return n_true/n    

def normalizacao01(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x-minx)/(maxx-minx)


class MLP():

    def __init__(self,):
        self.hiden_size = 3
        self.output_size = 1

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _d_sigmoid(self, x):
        return self._sigmoid(x)*(1-self._sigmoid(x))

    def fit(self, X, y):
        n = X.shape[0]

        bias = (-1) * (np.ones(n))
        X = np.c_[bias, X]
    
        self.input_size = X.shape[1]

        self.w = np.random.rand(self.hiden_size, self.input_size)
        self.m = np.random.rand(self.output_size, self.hiden_size+1)
    
            # ui = self.w.dot(X.T)
            # zi = self._sigmoid(ui)
        
            # zi = np.array(zi).T
            # zi = np.c_[(-1)*(np.ones(n)), zi]
        
            # uk = self.m.dot(zi.T)
            # yk = self._sigmoid(uk)
            
            # ek = y - yk
        
            # duk = self._sigmoid(uk)            
            
            # deltak = ek * duk

            # for j in range(n):
            #     for i in range(self.hiden_size):
            #         sum_ = 0.
            #         for k in range(self.output_size):
            #             sum_ = sum_ + duk[k, j]*self.m[k, i]
            #         delta_i = self._d_sigmoid(ui[i, j])*sum_
            #         print(delta_i)
        
            # dui = []
            # for u in ui:
            #     dui.append([self._d_sigmoid(e) for e in u])
        
            # dui = np.array(dui)
            # dkm = deltak.T.dot(self.m)
            # deltai = dui.dot(dkm)
        
            # self.m = self.m + self._alpha * (deltak.dot(zi))
            # self.w = self.w + self._alpha * (deltai.dot(X.T))
    def predict(self, X):
        ui = self.w.dot(X.T)
        zi = self._sigmoid(ui)
        
        zi = np.array(zi).T
        zi = np.c_[(-1)*(np.ones(n)), zi]
        
        uk = self.m.dot(zi.T)
        yk = self._sigmoid(uk)


def test_MLP():
    data = np.loadtxt('./atividades/trabalho2/data/ex2data1.txt', delimiter=',')
    np.random.shuffle(data)

    n = data.shape[0]

    X = data[:, :2]
    X[:,0] = normalizacao01(X[:,0])
    X[:,1] = normalizacao01(X[:,1])
    y = data[:, -1]
    
    X_train = X[:int(0.7*n), :]
    X_test = X[int(0.7*n):, :]

    y_train = y[:int(0.7*n)]
    y_test = y[int(0.7*n):]

    clf = MLP()
    clf.fit(X_train, y_train)        
    y_pred_test = clf.predict(X_test)
    print(y_pred_test)
    print(acc(y_test, y_pred_test))

if __name__ == "__main__":
    test_MLP()
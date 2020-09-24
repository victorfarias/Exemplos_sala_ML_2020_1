from warnings import simplefilter
import numpy as np

def acc(y_real, y_pred):
    n = y_real.shape[0]
    n_true = np.count_nonzero(y_real == y_pred)
    return n_true/n    

def normalizacao01(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x-minx)/(maxx-minx)


class RegressaLogisticaGD():

    def __init__(self, n_iter, alpha):
        self.n_iter = n_iter
        self.alpha = alpha

    def logistic(self, x):
        return 1/(1+np.exp(-self.B @ x))

    def J(self, X, y):
        error = 0.
        for xi, yi in zip(X, y):
            yi_pred = self.logistic(xi)
            error = error + (-yi*np.log(yi_pred) - (1-yi)*np.log(1-yi_pred))
        return error

    def fit(self, X, y):
        n = X.shape[0]
        X = np.c_[np.ones(n), X]        
        m = X.shape[1]
        self.B = np.zeros(m)

        self.errors = []
        for _ in range(self.n_iter):
            g = np.zeros(m)
            for xi, yi in zip(X,y):
                y_pred = self.logistic(xi)
                ei = yi - y_pred
                g = g + ei*xi
            g = g*(self.alpha/n)
            # print(g)
            self.errors.append(self.J(X, y))
            self.B = self.B + g

    def predict(self, X):
        n = X.shape[0]
        X = np.c_[np.ones(n), X]
        y_pred = []
        for xi in X:
            yi_pred = 1 if self.logistic(xi)>=0.5 else 0
            y_pred.append(yi_pred)
        return np.array(y_pred)
            
def teste_log():
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

    clf = RegressaLogisticaGD(1000, 1.)
    clf.fit(X_train, y_train)    
    print('Modelo treinado!, error final: ' + str(clf.errors[-1]))
    y_pred_test = clf.predict(X_test)
    print(acc(y_test, y_pred_test))

class DiscriminanteQuadraticoGaussiano():

    def __init__(self):
        pass

    def fit(self, X, y):
        self.cs = np.unique(y)
        self.pcs = {}
        self.ucs = {}
        self.sigmas_c = {}
        n = X.shape[0]
        m = X.shape[1]

        for c in self.cs:
            # Calculo pc
            mask = (y==c)
            nc = np.count_nonzero(mask)
            pc = nc/n        
            self.pcs[c] = pc

            # Calculo uc
            Xc = X[mask, :]
            uc = np.mean(Xc, axis=0)
            self.ucs[c] = uc

            # Calculo sigma_c
            sigma_c = np.zeros((m,m))
            for xi in X:                
                p = (xi-uc)[np.newaxis].T @ (xi-uc)[np.newaxis]                
                sigma_c = sigma_c + p
            sigma_c = sigma_c/(n-1)
            self.sigmas_c[c] = sigma_c
    
    def predict(self, X):
        m = X.shape[1]
        y_pred = []
        for xi in X:
            ps = []
            for c in self.cs:
                sigma_c = self.sigmas_c[c]
                uc = self.ucs[c]
                pc = self.pcs[c]                

                det = np.linalg.det(sigma_c)
                f = 1/(np.sqrt(det*(2*np.pi)**m))
                exp = np.exp(-(1/2)*(xi-uc).T @ np.linalg.pinv(sigma_c) @ (xi-uc))
                pxc = f*exp                
                pcx = pxc*pc
                ps.append(pcx)
            y_pred.append(np.argmax(ps))
        return np.array(y_pred)
                      


def test_DQG():
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

    clf = DiscriminanteQuadraticoGaussiano()
    clf.fit(X_train, y_train)        
    y_pred_test = clf.predict(X_test)
    print(y_pred_test)
    print(acc(y_test, y_pred_test))

if __name__ == "__main__":
    test_DQG()
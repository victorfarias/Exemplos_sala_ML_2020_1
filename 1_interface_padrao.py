import numpy as np

# Método de regressão que retorna a média
class MeanRegressor():
    def __init__(self):
        pass

    def fit(self, X, y):
        mean = np.mean(y)
        self.mean = mean

    def predict(self, x):
        return self.mean

# Carregar o dado
X = [[1,2],[3,4]]
y = [2, 4]

# Treinar meu modelo usando fit()
regressor = MeanRegressor()
regressor.fit(X, y)

# Fazer testes/predicoes usando predict()
print(regressor.predict([4.,5.,6.]))

# Medir erro
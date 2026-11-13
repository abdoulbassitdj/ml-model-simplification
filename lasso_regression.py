import numpy as np

class RegressionLasso():
    def __init__(self, alpha=0.001, n_iter=50, lmbda=1.0):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lmbda = lmbda  # Force de la régularisation Lasso
        self.theta_ = None
        self.couts_ = []

    def fonction_cout(self, m, erreurs, theta):
        # Pénalité L1 (Lasso) : Somme des valeurs absolues
        penalite = (self.lmbda / m) * np.sum(np.abs(theta[1:]))
        return (1 / (2 * m)) * (erreurs.T @ erreurs).item() + penalite


    def fit(self, X, y):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        y = y.values.reshape(-1, 1)
        n = X.shape[1]
        self.theta_ = np.zeros((n, 1))

        for i in range(self.n_iter):
            erreurs = X @ self.theta_ - y
            self.couts_.append(self.fonction_cout(m, erreurs, self.theta_))
            
            # Vecteur contenant le SIGNE des thétas (+1 ou -1)
            reg_vector = np.sign(self.theta_)
            reg_vector[0] = 0  # On ne pénalise pas le biais
            
            # Le gradient utilise np.sign(theta) au lieu de theta
            gradient = (1 / m) * (X.T @ erreurs) + (self.lmbda / m) * reg_vector
            self.theta_ = self.theta_ - self.alpha * gradient


        # return self.theta_, self.couts_

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        return (X @ self.theta_).flatten()
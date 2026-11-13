import numpy as np

class RegressionRidge():
    def __init__(self, alpha=0.001, n_iter=50, lmbda=1.0):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lmbda = lmbda  # Force de la régularisation Ridge
        self.theta_ = None
        self.couts_ = []

    def fonction_cout(self, m, erreurs, theta):
        # On ignore le premier coefficient (le biais) dans la pénalité
        penalite = (self.lmbda / (2 * m)) * np.sum(theta[1:] ** 2)
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
            
            # Création du vecteur de régularisation : 0 pour le biais, lmbda pour le reste
            reg_vector = np.copy(self.theta_)
            reg_vector[0] = 0 
            
            # Formule mise à jour avec le terme (lmbda/m) * reg_vector
            gradient = (1 / m) * (X.T @ erreurs) + (self.lmbda / m) * reg_vector
            self.theta_ = self.theta_ - self.alpha * gradient

        # return self.theta_, self.couts_

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))
        return (X @ self.theta_).flatten()
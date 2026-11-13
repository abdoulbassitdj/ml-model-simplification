import pandas as pd
from correlation import corr_deux_listes

class FeatureCorrelation:
    def __init__(self, X: pd.DataFrame, seuil: float = 0.85):
        self.X = X
        self.seuil = seuil
        self.noms_colonnes = list(X.columns)
        
        # 1. Calcul de la matrice de corrélation
        num_features = len(self.noms_colonnes)
        matrice_brute = [[0.0] * num_features for _ in range(num_features)]
        
        for i in range(num_features):
            for j in range(num_features):
                # On transforme les colonnes de Pandas en listes simples
                col_i = X[self.noms_colonnes[i]].tolist()
                col_j = X[self.noms_colonnes[j]].tolist()
                # On utilise votre fonction personnalisée
                matrice_brute[i][j] = corr_deux_listes(col_i, col_j)
                
        # On stocke le résultat dans un DataFrame Pandas pour l'affichage Seaborn
        self.correlation_df = pd.DataFrame(matrice_brute, index=self.noms_colonnes, columns=self.noms_colonnes)
        
        # 2. Filtrage des variables trop corrélées
        exclus = set()
        for i in range(num_features):
            for j in range(i + 1, num_features):
                valeur_corr = abs(matrice_brute[i][j])
                if valeur_corr >= self.seuil:
                    exclus.add(self.noms_colonnes[j])
                    
        # On crée les listes finales
        self.variables_supprimees = list(exclus)
        self.variables_retenues = [col for col in self.noms_colonnes if col not in exclus]

    def obtenir_matrice(self) -> pd.DataFrame:
        return self.correlation_df
            
    def obtenir_X_filtre(self) -> pd.DataFrame:
        return self.X[self.variables_retenues]

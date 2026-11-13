# Regularization, Variable Selection & PCA

Educational project (Chapter 9) on simplifying linear regression models: **correlation-based variable selection**, **Ridge and Lasso regularization**, and **dimensionality reduction with PCA**. Ridge, Lasso and the correlation tools are implemented from scratch with NumPy and validated against scikit-learn.

## Project structure

| File | Description |
|---|---|
| `Chapitre9_Regularisation_ACP.ipynb` | Main notebook (French) |
| `Chapitre9_Regularisation_ACP_EN.ipynb` | Main notebook (English) |
| `data.csv` | Synthetic housing dataset (200 rows) |
| `correlation.py` | `corr_deux_listes(x, y)`: Pearson correlation between two lists |
| `feature_correlation.py` | `FeatureCorrelation`: correlation matrix and removal of redundant features |
| `ridge_regression.py` | `RegressionRidge`: L2-regularized regression (gradient descent) |
| `lasso_regression.py` | `RegressionLasso`: L1-regularized regression (subgradient descent) |

## Dataset

`data.csv` contains 200 samples with 11 features and a target:

- **Informative features:** `Superficie` (area, m²), `Chambres` (bedrooms), `Age`, `Distance_Centre` (distance to center)
- **Redundant feature:** `Superficie_pieds` (area in square feet, perfectly correlated with `Superficie`)
- **Noise features:** `Bruit_1` … `Bruit_6` (useless variables)
- **Target:** `Prix` (price)

## Notebook outline

0. Data preparation: train/test split, error function
1. Why simplify a model? Complexity, overfitting, generalization
2. Variable selection: Pearson correlation, redundant, useful and useless variables
3. Regularization: Ridge, Lasso, comparison with scikit-learn, regularization path, choosing λ by cross-validation, final comparison
4. Dimensionality reduction and PCA
5. Summary

## Modules

### `corr_deux_listes(x, y)`
Pearson correlation coefficient, returns `0.0` if either variance is zero.

### `FeatureCorrelation(X, seuil=0.85)`
Computes the full correlation matrix of a DataFrame with `corr_deux_listes` and drops, for each pair with `|r| >= seuil`, the later column.

- `obtenir_matrice()`: correlation matrix (DataFrame, ready for a Seaborn heatmap)
- `obtenir_X_filtre()`: DataFrame with the retained features
- Attributes: `variables_retenues`, `variables_supprimees`

### `RegressionRidge(alpha=0.001, n_iter=50, lmbda=1.0)` and `RegressionLasso(...)`
Linear regression trained by gradient descent; the bias term is not penalized.

| Model | Penalty | Gradient term |
|---|---|---|
| Ridge | `λ/(2m) · Σθ²` | `(λ/m) · θ` |
| Lasso | `λ/m · Σ|θ|` | `(λ/m) · sign(θ)` |

Parameters: `alpha` (learning rate), `n_iter` (iterations), `lmbda` (regularization strength).
Methods: `fit(X, y)` (`X` array-like, `y` pandas Series), `predict(X)`. Attributes: `theta_` (bias first), `couts_` (cost history).

```python
import pandas as pd
from feature_correlation import FeatureCorrelation
from ridge_regression import RegressionRidge

df = pd.read_csv("data.csv")
X, y = df.drop(columns="Prix"), df["Prix"]

X_filtered = FeatureCorrelation(X, seuil=0.85).obtenir_X_filtre()
# Standardize features before fitting (gradient descent is scale-sensitive)
X_scaled = (X_filtered - X_filtered.mean()) / X_filtered.std()

model = RegressionRidge(alpha=0.01, n_iter=500, lmbda=1.0)
model.fit(X_scaled.values, y)
print(model.predict(X_scaled.values)[:5])
```

## Installation

Python 3.9+ and:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Then run:

```bash
jupyter notebook Chapitre9_Regularisation_ACP_EN.ipynb
```

Keep all files in the same folder so the notebook can import the local modules.

## Notes

- Standardize features before using Ridge/Lasso; the penalty depends on feature scale.
- In lasso_regression.py, Lasso is implemented via subgradient descent—the method we were most familiar with—which drives coefficients close to zero without setting them to exact zero. For the class presentation and in the notebook, we also explored coordinate descent to fulfill the assignment requirements.

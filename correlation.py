def corr_deux_listes(x, y):
    n = len(x)
    moy_x, moy_y = sum(x)/n, sum(y)/n
    cov = sum((x[i] - moy_x) * (y[i] - moy_y) for i in range(n))
    var_x = sum((x[i] - moy_x) ** 2 for i in range(n))
    var_y = sum((y[i] - moy_y) ** 2 for i in range(n))
    return cov / ((var_x * var_y) ** 0.5) if var_x and var_y else 0.0
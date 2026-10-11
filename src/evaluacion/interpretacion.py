# Dueño: Julián
"""VIF, coeficientes y estabilidad entre folds. Días 4 y 12 de la guía."""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calcular_vif(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Factor de inflación de varianza (VIF) de cada columna numérica.

    El VIF de la variable j es 1 / (1 - R²_j), donde R²_j sale de regresar la
    variable j contra todas las demás. Un VIF de 1 significa que la variable no se
    explica con las otras; entre 5 y 10 hay multicolinealidad moderada, y por encima
    de 10 se considera severa (el artículo base reporta 16,16 y 23,18).

    Se agrega una constante a la matriz porque el R² de esa regresión auxiliar debe
    calcularse con intercepto; sin ella statsmodels usaría un R² sin centrar y el VIF
    saldría inflado. La constante se calcula, pero no se devuelve: no es un predictor.

    Parameters
    ----------
    df : DataFrame que contiene las columnas (solo entrenamiento: regla 3 del repo).
    columnas : nombres de las columnas numéricas a evaluar, en el orden deseado.

    Returns
    -------
    DataFrame con las columnas ``variable`` y ``vif``, en el mismo orden de ``columnas``.
    """
    X = df[columnas].astype(float)
    if X.isna().any().any():
        raise ValueError("calcular_vif no admite valores faltantes en las columnas dadas")

    # has_constant="add" fuerza la constante aunque alguna columna sea constante;
    # con el valor por defecto ("skip") no se agregaría y los índices quedarían corridos.
    X = sm.add_constant(X, has_constant="add")

    # La columna 0 es la constante: el VIF se calcula desde la columna 1.
    vifs = [variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])]
    return pd.DataFrame({"variable": list(columnas), "vif": vifs})

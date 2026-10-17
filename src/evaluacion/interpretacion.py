# Dueño: Julián

"""VIF, coeficientes y estabilidad entre folds. Días 2, 6 y 12 de la guía."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
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


# ---------------------------------------------------------------------------
# Coeficientes y estabilidad (día 6 de la guía; se usan en el notebook 04, día 12)
# ---------------------------------------------------------------------------
def _desenvolver(modelo_ajustado):
    """Devuelve el Pipeline interno ajustado, quitando el TransformedTargetRegressor."""
    return getattr(modelo_ajustado, "regressor_", modelo_ajustado)


def coeficientes(modelo_ajustado) -> pd.DataFrame:
    """Coeficientes del modelo lineal final, con el nombre legible de cada variable.

    Acepta el resultado de construir_pipeline() ya ajustado (con o sin
    TransformedTargetRegressor) o un Pipeline simple cuyo último paso tenga ``coef_``.

    Como el StandardScaler va antes del modelo, el coeficiente de un predictor numérico
    está "estandarizado": es el cambio en log(índice) cuando el predictor sube una
    desviación estándar. Así se pueden comparar entre sí aunque tengan unidades distintas
    (pesos, %, m², miles de USD). Las dummies (cat__) y covid no se escalan: su
    coeficiente es el cambio en log(índice) frente a la categoría de referencia.

    Returns
    -------
    DataFrame con ``variable``, ``coeficiente`` y ``tipo`` (categorica, numerica, binaria),
    ordenado como entra al modelo.
    """
    pipe = _desenvolver(modelo_ajustado)
    final = pipe[-1]
    if not hasattr(final, "coef_"):
        raise TypeError(f"{type(final).__name__} no tiene coeficientes (¿es KNN?)")
    # Con set_output("pandas") el modelo guarda los nombres; si no, se piden a los pasos
    # anteriores (preprocesador y selector), que saben qué columnas dejaron pasar.
    if hasattr(final, "feature_names_in_"):
        nombres = list(final.feature_names_in_)
    else:
        nombres = list(pipe[:-1].get_feature_names_out())
    coef = np.ravel(final.coef_)
    tipo = [
        "categorica" if n.startswith("cat__") else "binaria" if n.startswith("bin__")
        else "numerica"
        for n in nombres
    ]
    return pd.DataFrame({"variable": nombres, "coeficiente": coef, "tipo": tipo})


def coeficientes_por_fold(pipeline, df: pd.DataFrame, cv, objetivo: str) -> pd.DataFrame:
    """Ajusta una copia del pipeline en el entrenamiento de cada fold y junta sus coeficientes.

    Solo usa los meses de entrenamiento de cada fold (los de validación no se tocan), así
    que se puede correr antes de abrir la prueba.
    """
    tablas = []
    for i, (entrena, _) in enumerate(cv.split(df), start=1):
        parte = df.iloc[entrena]
        ajustado = clone(pipeline).fit(parte, parte[objetivo])
        tablas.append(coeficientes(ajustado).assign(fold=i))
    return pd.concat(tablas, ignore_index=True)


def estabilidad(resultados_por_fold: pd.DataFrame, tolerancia: float = 1e-10) -> pd.DataFrame:
    """Media ± desviación de cada coeficiente entre folds.

    Es la respuesta directa a "¿la regularización mejora la estabilidad?": si un
    coeficiente cambia mucho (o cambia de signo) según los meses con que se entrena, no
    se puede interpretar. Se compara la desviación de MLR completa, Ridge y Lasso.

    Parameters
    ----------
    resultados_por_fold : DataFrame con columnas ``fold``, ``variable`` y ``coeficiente``
        (lo que devuelve coeficientes_por_fold). Si una variable falta en un fold (porque
        el stepwise no la eligió), cuenta como coeficiente 0 en ese fold.
    tolerancia : por debajo de este valor absoluto el coeficiente se considera cero.

    Returns
    -------
    DataFrame por variable con: media, desviacion, cv_abs (desviación / |media|),
    folds_distinto_de_cero y signo_estable (True si nunca cambia de signo entre los folds
    en que es distinto de cero).
    """
    ancha = resultados_por_fold.pivot_table(
        index="variable", columns="fold", values="coeficiente", aggfunc="first", sort=False
    ).fillna(0.0)
    distinto_cero = ancha.abs() > tolerancia
    signos = np.sign(ancha.where(distinto_cero))
    tabla = pd.DataFrame({
        "media": ancha.mean(axis=1),
        "desviacion": ancha.std(axis=1, ddof=1),
        "folds_distinto_de_cero": distinto_cero.sum(axis=1),
        "signo_estable": signos.nunique(axis=1) <= 1,
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        tabla.insert(2, "cv_abs", tabla["desviacion"] / tabla["media"].abs())
    return tabla.reset_index()

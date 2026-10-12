# Dueño: Juanes
"""construir_pipeline(): une preprocesamiento, selector y modelo. Día 4 de la guía.

Es la pieza central de la regla de oro 2 ("nada aprende fuera del Pipeline"):

    TransformedTargetRegressor(            ← modela log(y) y devuelve ŷ en escala original
        Pipeline([
            ("prep",   crear_preprocesador()),   ← one-hot, log de vivienda, escalado
            ("sel",    selector o "passthrough"),← StepwiseSelector de Julián o nada
            ("modelo", modelo),                  ← LinearRegression, Ridge, Lasso o KNN
        ]),
        func=np.log, inverse_func=np.exp)

Cuando GridSearchCV o evaluar_cv lo ajustan con los meses de entrenamiento de un fold,
TODO (categorías, medias, variables elegidas, coeficientes) se aprende solo con esos meses.

Nombres de los hiperparámetros para las rejillas de candidatos.py (Julián):
    con log_objetivo=True:   "regressor__modelo__alpha", "regressor__modelo__n_neighbors"
    con log_objetivo=False:  "modelo__alpha", "modelo__n_neighbors"
Se puede usar la función prefijo_parametros() para no escribirlos a mano.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.config import LOG_OBJETIVO
from src.modelos.preprocesamiento import crear_preprocesador


def construir_pipeline(modelo, selector=None, log_objetivo: bool = LOG_OBJETIVO,
                       columnas_log: list[str] | None = None):
    """Pipeline([prep, sel, modelo]), envuelto en TransformedTargetRegressor si log_objetivo.

    El paso "sel" siempre existe ("passthrough" si no hay selector) para que los nombres de
    los pasos sean iguales en los 5 modelos y las rejillas no cambien de un modelo a otro.
    """
    pipe = Pipeline([
        ("prep", crear_preprocesador(columnas_log)),
        ("sel", selector if selector is not None else "passthrough"),
        ("modelo", modelo),
    ])
    if not log_objetivo:
        return pipe
    # Decisión del día 1: el índice es L × F × η, en logaritmo los factores se suman.
    # El regresor aprende log(y); predict() aplica exp y devuelve el índice en su escala,
    # que es donde se calculan MAPE, MAD y MSD.
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


def prefijo_parametros(log_objetivo: bool = LOG_OBJETIVO) -> str:
    """Prefijo de los hiperparámetros del modelo en la rejilla de GridSearchCV."""
    return "regressor__modelo__" if log_objetivo else "modelo__"

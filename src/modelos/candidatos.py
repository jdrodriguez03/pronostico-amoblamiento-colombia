# Dueño: Julián
"""Los 5 modelos candidatos y sus rejillas de hiperparámetros. Día 4 de la guía.

CANDIDATOS = {nombre: (modelo, selector, rejilla)}

| Nombre       | Qué es                                | Para qué está                    |
| ------------ | ------------------------------------- | -------------------------------- |
| mlr_stepwise | Regresión lineal + stepwise (α 0,25)  | Reproduce el método del artículo |
| mlr_completa | Regresión lineal con todo             | Ver la multicolinealidad         |
| ridge        | Penalización L2: encoge coeficientes  | Estabilizar los correlacionados  |
| lasso        | Penalización L1: lleva algunos a cero | Selección "suave" vs stepwise    |
| knn          | Promedio de los k casos más parecidos | Contraste no lineal (rúbrica)    |

Las rejillas usan el nombre corto del hiperparámetro ("alpha", "n_neighbors"). El prefijo
que pide GridSearchCV según el Pipeline de Juanes ("regressor__modelo__" con el objetivo
en logaritmo) lo agrega rejilla_pipeline(), para no escribirlo a mano en cada modelo.

Uso desde scripts/entrenar.py (Juanes):

    from src.modelos.candidatos import CANDIDATOS, construir_candidato
    for nombre in CANDIDATOS:
        pipeline, rejilla = construir_candidato(nombre)
        GridSearchCV(pipeline, rejilla, cv=CVTemporalPorMes(), ...)
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.config import ALFA_STEPWISE, LOG_OBJETIVO
from src.modelos.stepwise import StepwiseSelector

# Las dummies que crea el ColumnTransformer de Juanes empiezan por "cat__" (mes,
# departamento y línea). Son la estacionalidad y los efectos fijos: siempre quedan.
PREFIJOS_FORZADOS = ["cat__"]

CANDIDATOS: dict = {
    # Sin rejilla: alfa = 0,25 es una decisión del artículo, no un hiperparámetro que
    # ajustemos (si lo ajustáramos ya no reproduciríamos su método).
    "mlr_stepwise": (
        LinearRegression(),
        StepwiseSelector(ALFA_STEPWISE, ALFA_STEPWISE, forzadas=PREFIJOS_FORZADOS),
        {},
    ),
    "mlr_completa": (LinearRegression(), None, {}),
    # 13 valores de 0,001 a 1000 (4 por década en escala log): cubre desde "casi MLR"
    # hasta "casi todo encogido a cero".
    "ridge": (Ridge(), None, {"alpha": np.logspace(-3, 3, 13)}),
    # Lasso necesita alfas más pequeños que Ridge para no apagar todo: 0,0001 a 10.
    # max_iter alto porque con predictores correlacionados converge lento.
    "lasso": (Lasso(max_iter=20000), None, {"alpha": np.logspace(-4, 1, 11)}),
    "knn": (
        KNeighborsRegressor(),
        None,
        {"n_neighbors": [3, 5, 10, 20, 40], "weights": ["uniform", "distance"]},
    ),
}


def prefijo(log_objetivo: bool = LOG_OBJETIVO) -> str:
    """Igual a prefijo_parametros() de src/modelos/pipeline.py (Juanes)."""
    return "regressor__modelo__" if log_objetivo else "modelo__"


def rejilla_pipeline(nombre: str, log_objetivo: bool = LOG_OBJETIVO) -> dict:
    """Rejilla del candidato con los nombres completos que entiende GridSearchCV."""
    _, _, rejilla = CANDIDATOS[nombre]
    return {f"{prefijo(log_objetivo)}{k}": list(v) for k, v in rejilla.items()}


def construir_candidato(nombre: str, log_objetivo: bool = LOG_OBJETIVO):
    """(pipeline sin ajustar, rejilla) listo para GridSearchCV.

    Clona el modelo y el selector para que dos llamadas no compartan el mismo objeto.
    La salida del Pipeline interno se pide como DataFrame (set_output) para que cada
    columna conserve su nombre: el stepwise reconoce las forzadas por el prefijo "cat__"
    y coeficientes() de interpretacion.py puede mostrar nombres legibles.
    """
    from src.modelos.pipeline import construir_pipeline  # de Juanes

    modelo, selector, _ = CANDIDATOS[nombre]
    pipe = construir_pipeline(
        clone(modelo),
        clone(selector) if selector is not None else None,
        log_objetivo=log_objetivo,
    )
    interno = pipe.regressor if log_objetivo else pipe
    interno.set_output(transform="pandas")
    return pipe, rejilla_pipeline(nombre, log_objetivo)

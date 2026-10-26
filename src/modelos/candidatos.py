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

# Opciones del Pipeline de Juanes que cambian según el modelo. KNN usa el one-hot sin
# drop: con drop="first" la categoría de referencia (por ejemplo enero) queda a distancia 1
# de todas las demás, mientras que dos categorías que no son la referencia quedan a √2
# entre sí, y KNN vería a la referencia "más cerca" de todo sin ninguna razón. La regresión
# lineal sí necesita el drop (si no, las dummies son colineales con el intercepto), así que
# los demás modelos se quedan con el valor por defecto del Pipeline.
OPCIONES_PIPELINE: dict = {
    "knn": {"drop_categorica": None},
}

CANDIDATOS: dict = {
    # Sin rejilla: alfa = 0,25 es una decisión del artículo, no un hiperparámetro que
    # ajustemos (si lo ajustáramos ya no reproduciríamos su método).
    "mlr_stepwise": (
        LinearRegression(),
        StepwiseSelector(ALFA_STEPWISE, ALFA_STEPWISE, forzadas=PREFIJOS_FORZADOS),
        {},
    ),
    "mlr_completa": (LinearRegression(), None, {}),
    # 17 valores de 0,001 a 100.000 (2 por década en escala log): cubre desde "casi MLR"
    # hasta "casi todo encogido a cero". Antes llegaba solo a 1000, y en la exploración del
    # 26 de octubre el mínimo de validación caía justo en ese borde: con la rejilla cortada
    # no se podía saber si el mínimo real estaba más allá. Con el tope en 100.000 la curva
    # vuelve a subir después del mínimo y se ve que es un mínimo de verdad.
    "ridge": (Ridge(), None, {"alpha": np.logspace(-3, 5, 17)}),
    # Lasso necesita alfas más pequeños que Ridge para no apagar todo: 0,0001 a 10.
    # max_iter alto porque con predictores correlacionados converge lento.
    "lasso": (Lasso(max_iter=20000), None, {"alpha": np.logspace(-4, 1, 11)}),
    # k desde 1: en la exploración del 26 de octubre el mínimo caía en k = 3, el valor más
    # pequeño de la rejilla anterior (el mismo problema de borde que Ridge). Con 1 y 2 se ve
    # que la curva vuelve a subir por la izquierda (k = 1 memoriza el entrenamiento).
    "knn": (
        KNeighborsRegressor(),
        None,
        {"n_neighbors": [1, 2, 3, 5, 10, 20, 40], "weights": ["uniform", "distance"]},
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
        **OPCIONES_PIPELINE.get(nombre, {}),
    )
    interno = pipe.regressor if log_objetivo else pipe
    interno.set_output(transform="pandas")
    return pipe, rejilla_pipeline(nombre, log_objetivo)

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

Entrega final (lunes 9 de noviembre):
- construir_pipeline recibe predictores=None y lo deja pasar a crear_preprocesador
  (contrato de la sección 3). Con None todo queda como en la Entrega 2.
- preparar_datos(df, rezagos) deja la base lista para una de las dos especificaciones
  ("rez1", la de la Entrega 2, o "reales", la de Julián) y devuelve también la lista de
  predictores. Es el ÚNICO lugar donde se decide qué columnas entran, para que
  entrenar.py, evaluar_prueba.py y final.py no puedan quedar con especificaciones
  distintas.
- candidato_con_predictores(nombre, predictores) es construir_candidato de Julián con los
  predictores que se le indiquen (su función no tiene ese parámetro y no se cambia).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.config import LOG_OBJETIVO, predictores_modelo
from src.features.rezagos import REZAGOS_REALES, agregar_rezagos_reales, columnas_predictoras
from src.modelos.preprocesamiento import crear_preprocesador

# Las dos especificaciones de predictores que se comparan en la entrega final
REZAGOS = ("reales", "rez1")
# Orden fijo de las filas: ver preparar_datos()
ORDEN_FILAS = ["fecha", "departamento", "linea_codigo"]


def construir_pipeline(modelo, selector=None, log_objetivo: bool = LOG_OBJETIVO,
                       columnas_log: list[str] | None = None,
                       drop_categorica: str | None = "first",
                       predictores: list[str] | None = None):
    """Pipeline([prep, sel, modelo]), envuelto en TransformedTargetRegressor si log_objetivo.

    El paso "sel" siempre existe ("passthrough" si no hay selector) para que los nombres de
    los pasos sean iguales en los 5 modelos y las rejillas no cambien de un modelo a otro.

    drop_categorica solo se deja pasar a crear_preprocesador(): "first" para los modelos
    lineales y None para KNN (la explicación está en crear_preprocesador). Va al final y con
    valor por defecto para que las llamadas que ya existen sigan funcionando igual.

    predictores también se deja pasar tal cual: None = config.predictores_modelo().
    """
    pipe = Pipeline([
        ("prep", crear_preprocesador(columnas_log, drop_categorica=drop_categorica,
                                     predictores=predictores)),
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


def preparar_datos(df: pd.DataFrame, rezagos: str = "reales") -> tuple[pd.DataFrame, list[str]]:
    """(base lista para el modelo, predictores numéricos) según la especificación.

    rezagos="reales" (la especificación final, corrección 2 de la profesora):
        aplica agregar_rezagos_reales de Julián, que agrega icc_rez2, ..., area_vivienda_m2_rez3
        mirando solo al pasado de cada serie departamento × línea, y descarta los primeros
        meses que no tienen su dato rezagado. Como cargar_entrenamiento ya empieza en febrero
        de 2019 y el rezago mayor es 3, el entrenamiento empieza en mayo de 2019 (75 meses
        en vez de 78). Los 5 folds validan los mismos 30 meses de antes (feb 2023 – jul
        2025), porque CVTemporalPorMes toma los folds desde el final.
    rezagos="rez1" (la especificación de la Entrega 2, solo para medir cuánto cuesta ser
        realistas): no agrega nada, usa las columnas _rez1 que ya trae la base.

    En los dos casos ordena las filas por fecha, departamento y línea con un ordenamiento
    estable. Por qué (decisión del 27 de octubre, que quedó pendiente): las 56 filas de un
    mes comparten los predictores nacionales y empatan en distancia en KNN, así que su
    resultado dependía del orden en que llegaban las filas, y ese orden podía cambiar entre
    computadores (8,70 % a 9,36 % en el central). Con un orden fijo KNN da lo mismo en
    cualquier PC. Ridge, MLR y las líneas base no dependen del orden. Se hace aquí y no en
    cargar.py porque cargar.py es de Aleja.

    Ejemplo: preparar_datos(cargar_entrenamiento("central")) → (4.200 filas, 5 nombres).
    """
    if rezagos not in REZAGOS:
        raise ValueError(f"rezagos debe ser uno de {REZAGOS}, no {rezagos!r}")
    if rezagos == "reales":
        datos, predictores = agregar_rezagos_reales(df, REZAGOS_REALES), columnas_predictoras()
    else:
        datos, predictores = df, predictores_modelo()
    faltan = [c for c in predictores if c not in datos.columns]
    if faltan:
        raise KeyError(f"a la base le faltan los predictores {faltan}")
    datos = datos.sort_values(ORDEN_FILAS, kind="stable").reset_index(drop=True)
    return datos, predictores


def candidato_con_predictores(nombre: str, predictores: list[str] | None = None,
                              log_objetivo: bool = LOG_OBJETIVO):
    """(pipeline sin ajustar, rejilla) de un candidato de candidatos.py con esos predictores.

    Hace lo mismo que construir_candidato() de Julián (clona modelo y selector, aplica
    OPCIONES_PIPELINE, como drop_categorica=None para KNN, y pide salida en DataFrame), con
    la única diferencia de que deja pasar predictores al preprocesador. Con predictores=None
    el resultado es idéntico a construir_candidato(nombre): lo comprueba
    tests/test_pipeline.py, para que las dos funciones no se separen sin que nadie lo note.
    """
    from src.modelos.candidatos import CANDIDATOS, OPCIONES_PIPELINE, rejilla_pipeline

    modelo, selector, _ = CANDIDATOS[nombre]
    pipe = construir_pipeline(
        clone(modelo),
        clone(selector) if selector is not None else None,
        log_objetivo=log_objetivo,
        predictores=predictores,
        **OPCIONES_PIPELINE.get(nombre, {}),
    )
    interno = pipe.regressor if log_objetivo else pipe
    interno.set_output(transform="pandas")
    return pipe, rejilla_pipeline(nombre, log_objetivo)

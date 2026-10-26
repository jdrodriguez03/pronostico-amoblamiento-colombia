# Dueño: Juanes
"""Evaluación con validación cruzada y scorer del MAPE nacional. Día 5 de la guía.

Plan de cierre: lunes 26, tarea 1 (camino crítico). Dos funciones:

- scorer_mape_nacional(estimador, X, y): lo que GridSearchCV usa para elegir
  hiperparámetros. El modelo predice las 56 series del panel (7 departamentos × 8
  líneas), pero el objetivo del trabajo, igual que en el artículo, es la serie NACIONAL
  de la línea 8 (electrodomésticos y muebles). Por eso el scorer:
      1. predice todas las filas de X,
      2. agrega la línea 8 a nivel nacional, con el promedio ponderado de los
         departamentos (agregar_nacional, de Aleja), tanto lo observado como lo predicho,
      3. calcula el MAPE entre las dos series nacionales (un valor por mes),
      4. devuelve −MAPE, porque scikit-learn siempre MAXIMIZA el score: el mejor modelo
         es el de menor MAPE, es decir, el de mayor −MAPE.

- evaluar_cv(pipeline, df, cv): ajusta y evalúa fold por fold y devuelve una tabla con
  una fila por fold. Sirve para las líneas base (que no tienen rejilla y no pasan por
  GridSearchCV) y para revisar cualquier modelo a mano.

Por qué agregar ANTES de calcular el MAPE (y no promediar el MAPE de cada serie): el
artículo reporta el error de la serie nacional. Los errores de departamentos distintos se
compensan en parte al sumar, así que el MAPE nacional suele ser menor que el del panel.
Por eso se reportan los dos y nunca se comparan entre sí.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import LINEA_OBJETIVO_CODIGO, OBJETIVO
from src.evaluacion.agregacion import agregar_nacional
from src.evaluacion.metricas import mad, mape, msd, rmse
from src.modelos.lineas_base import REZAGO_ESTACIONAL, IngenuoEstacional

# Columnas que necesita agregar_nacional para saber de qué serie y mes es cada fila
_CLAVES = ["fecha", "departamento", "linea_codigo"]


def _series_nacionales(X: pd.DataFrame, y, yhat) -> tuple[pd.Series, pd.Series]:
    """(observado, predicho) de la línea objetivo a nivel nacional, un valor por mes."""
    faltan = [c for c in _CLAVES if c not in X.columns]
    if faltan:
        raise ValueError(f"a X le faltan las columnas {faltan} para agregar a nivel nacional")
    datos = X[_CLAVES].copy()
    datos["y"] = np.asarray(y, dtype=float)
    datos["yhat"] = np.asarray(yhat, dtype=float)
    observado = agregar_nacional(datos, "y", LINEA_OBJETIVO_CODIGO)
    predicho = agregar_nacional(datos, "yhat", LINEA_OBJETIVO_CODIGO)
    return observado, predicho


def scorer_mape_nacional(estimador, X, y) -> float:
    """Predice, agrega la línea de muebles a nivel nacional y devuelve −MAPE.

    Tiene la firma (estimador, X, y) que GridSearchCV acepta como scoring: se pasa la
    función tal cual, scoring=scorer_mape_nacional, sin make_scorer.
    """
    observado, predicho = _series_nacionales(X, y, estimador.predict(X))
    return -mape(observado, predicho)


def _filas_con_t_menos_12(df: pd.DataFrame) -> np.ndarray:
    """Máscara de las filas cuyo mes t−12 está dentro del mismo df.

    El ingenuo estacional no puede predecir los primeros 12 meses del entrenamiento
    (no tiene su año anterior). Para su MAPE de entrenamiento se usan solo los meses
    desde el mes 13; para los demás modelos se usan todos.
    """
    fechas = pd.to_datetime(df["fecha"])
    return (fechas >= fechas.min() + pd.DateOffset(months=REZAGO_ESTACIONAL)).to_numpy()


def evaluar_cv(pipeline, df, cv):
    """DataFrame fold × métrica: MAPE, MAD, MSD y RMSE nacionales y MAPE del panel.

    Para cada fold de cv:
      - clona el pipeline (queda "en blanco") y lo ajusta SOLO con los meses de
        entrenamiento del fold: nada de validación entra al ajuste;
      - predice los meses de validación y calcula:
          mape, mad, msd, rmse  → serie nacional de la línea 8 (como el artículo)
          mape_panel            → todas las filas del panel (56 series)
      - calcula también mape_entrena: el MAPE nacional sobre los mismos meses con que se
        ajustó. Comparado con mape dice si el modelo subajusta (los dos altos) o
        sobreajusta (entrenamiento bajo y validación alta).
    """
    y = df[OBJETIVO].to_numpy(dtype=float)
    filas = []
    for k, (entrena, valida) in enumerate(cv.split(df), start=1):
        ent, val = df.iloc[entrena], df.iloc[valida]
        modelo = clone(pipeline).fit(ent, y[entrena])

        pred_val = modelo.predict(val)
        obs_nac, pred_nac = _series_nacionales(val, y[valida], pred_val)

        # Entrenamiento: el ingenuo estacional solo desde el mes 13 (ver arriba)
        if isinstance(modelo, IngenuoEstacional):
            ent_eval = ent[_filas_con_t_menos_12(ent)]
        else:
            ent_eval = ent
        y_ent_eval = ent_eval[OBJETIVO].to_numpy(dtype=float)
        obs_ent, pred_ent = _series_nacionales(ent_eval, y_ent_eval, modelo.predict(ent_eval))

        fechas_val = pd.to_datetime(val["fecha"])
        filas.append({
            "fold": k,
            "valida_desde": fechas_val.min(),
            "valida_hasta": fechas_val.max(),
            "mape": mape(obs_nac, pred_nac),
            "mad": mad(obs_nac, pred_nac),
            "msd": msd(obs_nac, pred_nac),
            "rmse": rmse(obs_nac, pred_nac),
            "mape_panel": mape(y[valida], pred_val),
            "mape_entrena": mape(obs_ent, pred_ent),
        })
    return pd.DataFrame(filas)

# Dueño: Juanes
"""Líneas base: predictor de la media e ingenuo estacional. Día 3 de la guía.

Una línea base es el pronóstico más simple razonable. Sirve de vara: un modelo que no
la supera no aporta nada, por sofisticado que sea. Se usan dos:

- PredictorMedia: para cada serie (departamento × línea) pronostica el promedio de lo
  que vio en entrenamiento. Ignora la estacionalidad y la tendencia: es la vara baja.
- IngenuoEstacional: pronostica el mes t con el valor de la misma serie en t−12 (el mismo
  mes del año anterior). Captura la estacionalidad y el nivel reciente sin estimar nada:
  es la vara exigente, la que de verdad deben superar los modelos.

Las dos siguen la interfaz de scikit-learn (fit / predict), así que evaluar_cv y
GridSearchCV las tratan igual que a cualquier modelo. A diferencia del Pipeline, NO usan
los predictores económicos: solo leen las columnas que identifican la serie y el mes
(departamento, linea_codigo y fecha), por eso reciben X como DataFrame completo.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# Columnas que identifican una serie del panel (7 × 8 = 56 series)
SERIE = ["departamento", "linea_codigo"]
REZAGO_ESTACIONAL = 12  # meses


def _revisar_columnas(X, columnas: list[str]) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("las líneas base necesitan X como DataFrame (con fecha y la serie)")
    faltan = [c for c in columnas if c not in X.columns]
    if faltan:
        raise ValueError(f"a X le faltan las columnas {faltan}")


class PredictorMedia(BaseEstimator, RegressorMixin):
    """Predice el promedio de entrenamiento de cada serie departamento x línea."""

    def fit(self, X, y):
        _revisar_columnas(X, SERIE)
        datos = X[SERIE].copy()
        datos["y"] = np.asarray(y, dtype=float)
        # Un promedio por serie, calculado SOLO con las filas de entrenamiento del fold
        self.medias_ = datos.groupby(SERIE)["y"].mean()
        return self

    def predict(self, X):
        _revisar_columnas(X, SERIE)
        claves = pd.MultiIndex.from_frame(X[SERIE])
        pred = self.medias_.reindex(claves).to_numpy()
        if np.isnan(pred).any():
            raise ValueError("hay series en X que no estaban en el entrenamiento")
        return pred


class IngenuoEstacional(BaseEstimator, RegressorMixin):
    """Predice para cada fila el valor de la misma serie 12 meses antes."""

    def fit(self, X, y):
        _revisar_columnas(X, [*SERIE, "fecha"])
        datos = X[[*SERIE, "fecha"]].copy()
        datos["fecha"] = pd.to_datetime(datos["fecha"])
        datos["y"] = np.asarray(y, dtype=float)
        # "Memoria" del modelo: el valor observado de cada serie en cada mes de entrenamiento
        self.historia_ = datos.set_index([*SERIE, "fecha"])["y"]
        return self

    def predict(self, X):
        _revisar_columnas(X, [*SERIE, "fecha"])
        claves = X[[*SERIE, "fecha"]].copy()
        claves["fecha"] = (
            pd.to_datetime(claves["fecha"]) - pd.DateOffset(months=REZAGO_ESTACIONAL)
        )
        pred = self.historia_.reindex(pd.MultiIndex.from_frame(claves)).to_numpy()
        if np.isnan(pred).any():
            # Nunca debería pasar con CVTemporalPorMes: la validación dura 6 meses y el
            # mes t−12 siempre queda en entrenamiento (ver docs/validacion.md)
            raise ValueError("a alguna fila le falta su mes t−12 en el entrenamiento")
        return pred

# Dueño: Juanes
"""Líneas base: predictor de la media e ingenuo estacional. Día 3 de la guía."""

from sklearn.base import BaseEstimator, RegressorMixin


class PredictorMedia(BaseEstimator, RegressorMixin):
    """Predice el promedio de entrenamiento de cada serie departamento x línea."""

    def fit(self, X, y):
        raise NotImplementedError("Juanes, día 3")

    def predict(self, X):
        raise NotImplementedError("Juanes, día 3")


class IngenuoEstacional(BaseEstimator, RegressorMixin):
    """Predice para cada fila el valor de la misma serie 12 meses antes."""

    def fit(self, X, y):
        raise NotImplementedError("Juanes, día 3")

    def predict(self, X):
        raise NotImplementedError("Juanes, día 3")

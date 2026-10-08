# Dueño: Julián
"""StepwiseSelector: selección stepwise bidireccional (alfa = 0,25, como el artículo).

Compatible con scikit-learn para ir dentro del Pipeline y ajustarse en cada fold.
Día 6 de la guía.
"""

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from src.config import ALFA_STEPWISE


class StepwiseSelector(BaseEstimator, SelectorMixin):
    def __init__(self, alfa_entrada=ALFA_STEPWISE, alfa_salida=ALFA_STEPWISE, forzadas=None):
        self.alfa_entrada = alfa_entrada
        self.alfa_salida = alfa_salida
        self.forzadas = forzadas

    def fit(self, X, y):
        raise NotImplementedError("Julián, día 6")

    def _get_support_mask(self):
        raise NotImplementedError("Julián, día 6")

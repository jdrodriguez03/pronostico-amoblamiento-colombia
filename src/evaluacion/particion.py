# Dueño: Juanes
"""Validación cruzada temporal por mes (ventana creciente). Día 2 de la guía.

Las ventanas exactas están en docs/validacion.md.
"""

from src.config import CV_FOLDS, CV_MESES_VALIDACION


class CVTemporalPorMes:
    """Splitter compatible con GridSearchCV que agrupa las filas por la columna fecha."""

    def __init__(self, n_splits: int = CV_FOLDS, meses_validacion: int = CV_MESES_VALIDACION):
        self.n_splits = n_splits
        self.meses_validacion = meses_validacion

    def split(self, X, y=None, groups=None):
        raise NotImplementedError("Juanes, día 2")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

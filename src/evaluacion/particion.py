# Dueño: Juanes
"""Validación cruzada temporal por mes (ventana creciente). Día 2 de la guía.

Las ventanas exactas están en docs/validacion.md. Con los 78 meses de entrenamiento
(febrero de 2019 a julio de 2025), 5 folds y 6 meses de validación por fold:

    fold 1: entrena 48 meses (feb 2019 – ene 2023), valida feb 2023 – jul 2023
    ...
    fold 5: entrena 72 meses (feb 2019 – ene 2025), valida feb 2025 – jul 2025

Por qué así (regla de oro 1):
- Se parte por MES y no por fila: las 56 filas de un mes (7 departamentos × 8 líneas)
  comparten los mismos predictores nacionales. Si unas quedaran en entrenamiento y otras
  en validación, el modelo "vería" ese mes y el error de validación saldría optimista.
- Ventana creciente: cada fold entrena con todo el pasado disponible y valida con los
  6 meses siguientes, igual que se pronostica en la práctica. Nunca se valida con el pasado.
- La validación siempre es posterior al entrenamiento, así que el ingenuo estacional
  (que usa el mes t−12) siempre encuentra su dato en entrenamiento.

Uso:
    from src.evaluacion.particion import CVTemporalPorMes
    cv = CVTemporalPorMes()
    GridSearchCV(pipeline, rejilla, cv=cv, ...)   # o for entrena, valida in cv.split(df)
    cv.ventanas(df)                              # tabla legible de los folds
"""

import numpy as np
import pandas as pd

from src.config import CV_FOLDS, CV_MESES_VALIDACION, INICIO_PRUEBA


class CVTemporalPorMes:
    """Splitter compatible con GridSearchCV que agrupa las filas por la columna fecha.

    No hereda de las clases de scikit-learn: para que GridSearchCV lo acepte basta con
    que tenga los métodos ``split`` y ``get_n_splits`` (así funciona cualquier ``cv``).
    """

    def __init__(self, n_splits: int = CV_FOLDS, meses_validacion: int = CV_MESES_VALIDACION):
        self.n_splits = n_splits
        self.meses_validacion = meses_validacion

    # ------------------------------------------------------------------
    # Lógica interna: decidir qué MESES van a cada lado (no qué filas)
    # ------------------------------------------------------------------
    def _meses(self, X) -> tuple[pd.Series, np.ndarray]:
        """Devuelve la columna fecha de X y la lista ordenada de meses distintos."""
        if not isinstance(X, pd.DataFrame) or "fecha" not in X.columns:
            raise ValueError("CVTemporalPorMes necesita un DataFrame con la columna 'fecha'")
        fechas = pd.to_datetime(X["fecha"])
        # Regla de oro 3: la validación cruzada nunca debe recibir meses de la prueba
        if (fechas >= pd.Timestamp(INICIO_PRUEBA)).any():
            raise ValueError(
                f"X trae meses desde {INICIO_PRUEBA}: la prueba no entra a la validación "
                "cruzada. Usa cargar_entrenamiento()."
            )
        meses = np.sort(fechas.unique())
        minimo = (self.n_splits + 1) * self.meses_validacion
        if len(meses) < minimo:
            raise ValueError(
                f"Hay {len(meses)} meses y se necesitan al menos {minimo} "
                f"para {self.n_splits} folds de {self.meses_validacion} meses"
            )
        return fechas, meses

    def _meses_por_fold(self, meses: np.ndarray):
        """Para cada fold, (meses de entrenamiento, meses de validación)."""
        n = len(meses)
        for k in range(self.n_splits):
            # El primer mes de validación del fold k (k = 0 es el fold 1)
            inicio_val = n - (self.n_splits - k) * self.meses_validacion
            fin_val = inicio_val + self.meses_validacion
            yield meses[:inicio_val], meses[inicio_val:fin_val]

    # ------------------------------------------------------------------
    # Interfaz que usa scikit-learn
    # ------------------------------------------------------------------
    def split(self, X, y=None, groups=None):
        """Genera (índices de entrenamiento, índices de validación) por posición de fila."""
        fechas, meses = self._meses(X)
        valores = fechas.to_numpy()
        for meses_ent, meses_val in self._meses_por_fold(meses):
            entrena = np.flatnonzero(np.isin(valores, meses_ent))
            valida = np.flatnonzero(np.isin(valores, meses_val))
            yield entrena, valida

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    # ------------------------------------------------------------------
    # Ayuda para documentar y probar
    # ------------------------------------------------------------------
    def ventanas(self, X) -> pd.DataFrame:
        """Tabla con el rango de meses de cada fold (la de docs/validacion.md)."""
        _, meses = self._meses(X)
        filas = []
        for k, (ent, val) in enumerate(self._meses_por_fold(meses), start=1):
            filas.append({
                "fold": k,
                "entrena_desde": pd.Timestamp(ent[0]),
                "entrena_hasta": pd.Timestamp(ent[-1]),
                "meses_entrena": len(ent),
                "valida_desde": pd.Timestamp(val[0]),
                "valida_hasta": pd.Timestamp(val[-1]),
                "meses_valida": len(val),
            })
        return pd.DataFrame(filas)

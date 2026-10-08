# Dueño: Aleja
"""Simulación del componente eta(d, l, t): la única parte simulada del índice.

AR(1) en logaritmo con desviación estándar sigma y coeficiente PHI, ajustado para
que el promedio ponderado de los departamentos reproduzca la serie nacional.
Día 3 de la guía.
"""

import pandas as pd

from src.config import PHI, SEMILLA


def simular_eta(sigma: float, phi: float = PHI, semilla: int = SEMILLA) -> pd.DataFrame:
    """Devuelve columnas fecha, departamento, linea, eta (sin conservar todavía)."""
    raise NotImplementedError("Aleja, día 3")


def conservar(eta: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """Divide eta(d,l,t) entre la suma sobre d de w_d * F(d,t) * eta(d,l,t)."""
    raise NotImplementedError("Aleja, día 3")

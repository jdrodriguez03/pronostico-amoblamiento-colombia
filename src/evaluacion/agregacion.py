# Dueño: Aleja
"""Agregación del panel a la serie nacional de una línea (suma ponderada por PESOS_DEPTO).

Día 4 de la guía.
"""

import pandas as pd

from src.config import LINEA_OBJETIVO


def agregar_nacional(df: pd.DataFrame, columna: str, linea: str = LINEA_OBJETIVO) -> pd.Series:
    """Suma sobre d de w_d * columna, por mes, para la línea indicada."""
    raise NotImplementedError("Aleja, día 4")

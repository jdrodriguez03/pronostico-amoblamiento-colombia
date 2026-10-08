# Dueño: Aleja
"""Agregación del panel a la serie nacional de una línea.

Promedio ponderado de los 7 departamentos con config.PESOS_DEPTO (columna
peso_base2019 de componentes_regionales.csv). Por construcción, para el índice
real reproduce exactamente la serie nacional del DANE (lineas_nacionales.csv).
Día 2 de la guía.
"""

import pandas as pd

from src.config import LINEA_OBJETIVO_CODIGO


def agregar_nacional(df: pd.DataFrame, columna: str, linea_codigo: int = LINEA_OBJETIVO_CODIGO
                     ) -> pd.Series:
    """Suma sobre d de w_d * columna, por mes, para la línea indicada."""
    raise NotImplementedError("Aleja, día 2")

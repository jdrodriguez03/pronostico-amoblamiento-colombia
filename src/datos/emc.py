# Dueño: Aleja
"""Lectura de los anexos de la Encuesta Mensual de Comercio (EMC) del DANE.

Archivos en data/raw/dane_emc/. Día 2 de la guía.
"""

import pandas as pd


def leer_lineas_nacional() -> pd.DataFrame:
    """L(l, t): índice nacional de ventas reales por línea (anexo nacional, hoja 2.2).

    Devuelve columnas fecha, linea, L (8 líneas x 91 meses = 728 filas).
    """
    raise NotImplementedError("Aleja, día 2")


def leer_departamental() -> pd.DataFrame:
    """Índice del grupo CIIU 4741-4759 por dominio (anexo departamental, hoja 3.2).

    Devuelve columnas fecha, departamento, indice_depto (7 x 91 = 637 filas).
    """
    raise NotImplementedError("Aleja, día 2")


def factor_regional(departamental: pd.DataFrame) -> pd.DataFrame:
    """F(d, t) = índice del dominio / suma ponderada de los dominios (con PESOS_DEPTO).

    Devuelve columnas fecha, departamento, F.
    """
    raise NotImplementedError("Aleja, día 2")

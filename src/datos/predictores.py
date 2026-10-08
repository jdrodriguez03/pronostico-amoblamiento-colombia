# Dueño: Julián
"""Lectura de las fuentes de los predictores (data/raw/, ver data/raw/FUENTES.md).

Cada función devuelve una tabla larga con fecha (primer día del mes) y, si aplica,
ciudad o departamento. Días 2 y 3 de la guía.
"""

import pandas as pd


def leer_icc() -> pd.DataFrame:
    """Índice de Confianza del Consumidor de Fedesarrollo: nacional y 5 ciudades."""
    raise NotImplementedError("Julián, día 2")


def leer_elic() -> pd.DataFrame:
    """Área aprobada para vivienda (ELIC, DANE) por departamento."""
    raise NotImplementedError("Julián, día 2")


def leer_tasa_consumo() -> pd.DataFrame:
    """Tasa de interés de créditos de consumo (Banco de la República)."""
    raise NotImplementedError("Julián, día 2")


def leer_inflacion() -> pd.DataFrame:
    """Variación anual del IPC (DANE)."""
    raise NotImplementedError("Julián, día 2")


def leer_importaciones() -> pd.DataFrame:
    """Importaciones de muebles, suma de partidas 9401, 9403 y 9404 (DIAN)."""
    raise NotImplementedError("Julián, día 2")


def indicadora_pandemia() -> pd.DataFrame:
    """1 entre config.PANDEMIA_INICIO y config.PANDEMIA_FIN, 0 en otro caso."""
    raise NotImplementedError("Julián, día 2")


def construir_predictores() -> pd.DataFrame:
    """Une todo a nivel departamento x mes (637 filas): esquema.COLUMNAS_PREDICTORES."""
    raise NotImplementedError("Julián, día 3")

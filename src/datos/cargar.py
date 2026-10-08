# Dueño: Aleja
"""Carga de la base final ya partida en entrenamiento y prueba.

Regla de oro 3: cargar_prueba() SOLO se llama desde scripts/evaluar_prueba.py.
tests/test_fuga.py lo revisa.

Uso:
    from src.datos.cargar import cargar_entrenamiento
    df = cargar_entrenamiento("central")          # base real
    df = cargar_entrenamiento(fixture=True)       # base pequeña para pruebas rápidas
"""

import pandas as pd

from src.config import FECHA_INICIO, INICIO_PRUEBA, RUTA_FIXTURE, USAR_REZAGO, ruta_base
from src.datos.esquema import validar_base


def _leer(escenario: str, fixture: bool) -> pd.DataFrame:
    ruta = RUTA_FIXTURE if fixture else ruta_base(escenario)
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    validar_base(df)
    if USAR_REZAGO:
        # Enero de 2019 no tiene rezago de vivienda: se descarta en vez de imputarlo
        df = df[df["fecha"] > pd.Timestamp(FECHA_INICIO)]
    return df.reset_index(drop=True)


def cargar_entrenamiento(escenario: str = "central", fixture: bool = False) -> pd.DataFrame:
    """Meses anteriores a agosto de 2025 (con USAR_REZAGO, desde febrero de 2019)."""
    df = _leer(escenario, fixture)
    return df[df["fecha"] < pd.Timestamp(INICIO_PRUEBA)].reset_index(drop=True)


def cargar_prueba(escenario: str = "central") -> pd.DataFrame:
    """Agosto de 2025 a julio de 2026. Solo para scripts/evaluar_prueba.py."""
    df = _leer(escenario, fixture=False)
    return df[df["fecha"] >= pd.Timestamp(INICIO_PRUEBA)].reset_index(drop=True)

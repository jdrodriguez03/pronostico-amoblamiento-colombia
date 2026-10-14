"""Carga paneles y separa entrenamiento y prueba en el tiempo."""

import pandas as pd

from src.config import (
    ESCENARIOS,
    FECHA_INICIO,
    INICIO_PRUEBA,
    RUTA_FIXTURE,
    RUTA_PROCESSED,
    USAR_REZAGO,
    ruta_base,
)
from src.datos.esquema import validar_base


def _leer(escenario: str, fixture: bool) -> pd.DataFrame:
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario inválido: {escenario!r}")
    if fixture:
        df = pd.read_csv(RUTA_FIXTURE, parse_dates=["fecha"])
        validar_base(df)
    else:
        ruta_panel = RUTA_PROCESSED / f"panel_{escenario}.parquet"
        if ruta_panel.exists():
            df = pd.read_parquet(ruta_panel)
            df["fecha"] = pd.to_datetime(df["fecha"])
        else:
            # Compatibilidad transitoria con las bases finales CSV de Entrega 1.
            df = pd.read_csv(ruta_base(escenario), parse_dates=["fecha"])
            validar_base(df)
    if USAR_REZAGO:
        df = df[df["fecha"] > pd.Timestamp(FECHA_INICIO)]
    return df.sort_values("fecha").reset_index(drop=True)


def cargar_entrenamiento(escenario: str = "central", fixture: bool = False) -> pd.DataFrame:
    """Devuelve las fechas anteriores a INICIO_PRUEBA."""
    df = _leer(escenario, fixture)
    return df[df["fecha"] < pd.Timestamp(INICIO_PRUEBA)].reset_index(drop=True)


def cargar_prueba(escenario: str = "central") -> pd.DataFrame:
    """Devuelve INICIO_PRUEBA y las fechas posteriores disponibles."""
    df = _leer(escenario, fixture=False)
    return df[df["fecha"] >= pd.Timestamp(INICIO_PRUEBA)].reset_index(drop=True)

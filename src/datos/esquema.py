# Dueño: Aleja
"""El contrato de datos: qué columnas debe tener cada tabla y cómo se valida.

Es el acuerdo entre quienes construyen los datos (Aleja y Julián) y quienes modelan
(Juanes y Julián). Solo se cambia en un cierre de bloque.

Uso:
    from src.datos.esquema import validar_dataset
    validar_dataset(df)  # lanza ValueError con la lista de problemas, o devuelve df
"""

import pandas as pd

from src.config import (
    COLUMNAS_PROHIBIDAS,
    DEPARTAMENTOS,
    LINEAS,
    OBJETIVO,
    PREDICTORES_BIN,
    PREDICTORES_NUM,
)

# Columnas del panel (variable objetivo, sin predictores)
COLUMNAS_PANEL = ["fecha", "mes", "departamento", "linea", OBJETIVO]

# Columnas de la tabla de predictores (una fila por departamento y mes)
COLUMNAS_PREDICTORES = ["fecha", "departamento", *PREDICTORES_NUM, *PREDICTORES_BIN]

# Las 11 columnas del dataset que entra a los modelos
COLUMNAS_DATASET = [*COLUMNAS_PANEL, *PREDICTORES_NUM, *PREDICTORES_BIN]


def _revisar_comunes(df: pd.DataFrame, columnas: list[str], claves: list[str]) -> list[str]:
    """Revisiones que aplican a cualquier tabla del contrato. Devuelve la lista de problemas."""
    problemas = []

    prohibidas = [c for c in COLUMNAS_PROHIBIDAS if c in df.columns]
    if prohibidas:
        problemas.append(f"columnas prohibidas (fuga): {prohibidas}")

    faltan = [c for c in columnas if c not in df.columns]
    sobran = [c for c in df.columns if c not in columnas and c not in COLUMNAS_PROHIBIDAS]
    if faltan:
        problemas.append(f"faltan columnas: {faltan}")
    if sobran:
        problemas.append(f"sobran columnas: {sobran}")
    if faltan:
        return problemas  # sin las columnas no se puede revisar lo demás

    nulos = df[columnas].isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        problemas.append(f"valores nulos: {nulos.to_dict()}")

    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        problemas.append("la columna fecha no es de tipo fecha (usa pd.to_datetime)")
    elif not (df["fecha"].dt.day == 1).all():
        problemas.append("hay fechas que no son el primer día del mes")

    if "departamento" in claves:
        raros = set(df["departamento"]) - set(DEPARTAMENTOS)
        if raros:
            problemas.append(f"departamentos que no están en config.DEPARTAMENTOS: {sorted(raros)}")
    if "linea" in claves:
        raras = set(df["linea"]) - set(LINEAS)
        if raras:
            problemas.append(f"líneas que no están en config.LINEAS: {sorted(raras)}")

    repetidas = df.duplicated(subset=claves).sum()
    if repetidas:
        problemas.append(f"{repetidas} filas repetidas en {claves}")

    # Cuadrícula completa: cada combinación de las claves aparece exactamente una vez
    esperadas = 1
    for c in claves:
        esperadas *= df[c].nunique()
    if len(df) - repetidas != esperadas:
        problemas.append(
            f"la cuadrícula {' × '.join(claves)} está incompleta: "
            f"{len(df) - repetidas} filas únicas de {esperadas} esperadas"
        )
    return problemas


def _lanzar(nombre: str, problemas: list[str]) -> None:
    if problemas:
        detalle = "\n  - ".join(problemas)
        raise ValueError(f"{nombre} no cumple el contrato:\n  - {detalle}")


def validar_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Valida un panel_{escenario}: fecha × departamento × línea con la variable objetivo."""
    claves = ["fecha", "departamento", "linea"]
    problemas = _revisar_comunes(df, COLUMNAS_PANEL, claves)
    if "mes" in df.columns and "fecha" in df.columns and not problemas:
        if not (df["mes"] == df["fecha"].dt.month).all():
            problemas.append("la columna mes no coincide con el mes de fecha")
    if OBJETIVO in df.columns and (df[OBJETIVO] <= 0).any():
        problemas.append(f"{OBJETIVO} tiene valores menores o iguales a 0 (no admite logaritmo)")
    _lanzar("El panel", problemas)
    return df


def validar_predictores(df: pd.DataFrame) -> pd.DataFrame:
    """Valida predictores.parquet: una fila por departamento y mes."""
    problemas = _revisar_comunes(df, COLUMNAS_PREDICTORES, ["fecha", "departamento"])
    if "pandemia" in df.columns and not df["pandemia"].isin([0, 1]).all():
        problemas.append("pandemia solo puede valer 0 o 1")
    _lanzar("La tabla de predictores", problemas)
    return df


def validar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida un dataset_{escenario} (o el fixture): las 11 columnas del contrato."""
    claves = ["fecha", "departamento", "linea"]
    problemas = _revisar_comunes(df, COLUMNAS_DATASET, claves)
    if not problemas:
        if not (df["mes"] == df["fecha"].dt.month).all():
            problemas.append("la columna mes no coincide con el mes de fecha")
        if (df[OBJETIVO] <= 0).any():
            problemas.append(f"{OBJETIVO} tiene valores menores o iguales a 0")
        if not df["pandemia"].isin([0, 1]).all():
            problemas.append("pandemia solo puede valer 0 o 1")
        # Canario de fuga: ningún predictor puede ser idéntico al objetivo
        for c in PREDICTORES_NUM:
            if df[c].equals(df[OBJETIVO]):
                problemas.append(f"el predictor {c} es idéntico al objetivo (fuga)")
    _lanzar("El dataset", problemas)
    return df

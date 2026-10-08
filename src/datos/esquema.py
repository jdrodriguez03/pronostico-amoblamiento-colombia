# Dueño: Aleja
"""El contrato de datos: qué columnas tiene base_final_*.csv y cómo se valida.

Es el acuerdo entre la base construida (src/simulation, src/features) y el
modelamiento (src/modelos, src/evaluacion). Solo se cambia en un cierre de bloque.

Uso:
    from src.datos.esquema import validar_base
    validar_base(df)  # lanza ValueError con la lista de problemas, o devuelve df
"""

import pandas as pd

from src.config import (
    COLUMNAS_PROHIBIDAS,
    DEPARTAMENTOS,
    FECHA_INICIO,
    IDENTIFICADORES,
    LINEA_OBJETIVO_CODIGO,
    LINEAS,
    OBJETIVO,
    PREDICTORES_BIN,
    PREDICTORES_NUM,
    PREDICTORES_REZ1,
)

# Las 19 columnas de base_final_*.csv, en el orden en que las escribe predictores.py
COLUMNAS_BASE = [
    "fecha", "anio", "mes", "departamento", "linea_codigo", "linea", "es_muebles", OBJETIVO,
    "icc", "inflacion_anual", "tasa_consumo", "importaciones_muebles_usd_miles",
    "icc_rez1", "inflacion_anual_rez1", "tasa_consumo_rez1",
    "importaciones_muebles_usd_miles_rez1", "covid", "area_vivienda_m2", "area_vivienda_m2_rez1",
]
assert set(COLUMNAS_BASE) == {
    *IDENTIFICADORES, "mes", "departamento", "linea", OBJETIVO,
    *PREDICTORES_NUM, *PREDICTORES_REZ1, *PREDICTORES_BIN,
}

# Único vacío permitido: el rezago de vivienda en el primer mes (antes de 2019 la
# cobertura de ELIC no es comparable). Son 56 filas: 7 departamentos × 8 líneas.
VACIO_PERMITIDO = ("area_vivienda_m2_rez1", FECHA_INICIO)


def validar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Valida una base_final_{escenario} o un subconjunto de meses de ella."""
    problemas = []

    prohibidas = [c for c in COLUMNAS_PROHIBIDAS if c in df.columns]
    if prohibidas:
        problemas.append(f"columnas de auditoría usadas como predictores (fuga): {prohibidas}")
    faltan = [c for c in COLUMNAS_BASE if c not in df.columns]
    sobran = [c for c in df.columns if c not in COLUMNAS_BASE and c not in COLUMNAS_PROHIBIDAS]
    if faltan:
        problemas.append(f"faltan columnas: {faltan}")
    if sobran:
        problemas.append(f"sobran columnas: {sobran}")
    if faltan:
        _lanzar(problemas)

    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        problemas.append("la columna fecha no es de tipo fecha (usa parse_dates=['fecha'])")
        _lanzar(problemas)
    if not (df["fecha"].dt.day == 1).all():
        problemas.append("hay fechas que no son el primer día del mes")

    # Vacíos: solo el rezago de vivienda en enero de 2019
    col, mes = VACIO_PERMITIDO
    vacios = df.isna()
    vacios.loc[df["fecha"] == pd.Timestamp(mes), col] = False
    if vacios.any().any():
        conteo = vacios.sum()
        problemas.append(f"valores vacíos no permitidos: {conteo[conteo > 0].to_dict()}")

    # Coherencia de las columnas de identificación
    meses_ok = (df["mes"] == df["fecha"].dt.month).all()
    if not meses_ok or not (df["anio"] == df["fecha"].dt.year).all():
        problemas.append("anio o mes no coinciden con fecha")
    raros = set(df["departamento"]) - set(DEPARTAMENTOS)
    if raros:
        problemas.append(f"departamentos que no están en config.DEPARTAMENTOS: {sorted(raros)}")
    if not (df["linea_codigo"].map(LINEAS) == df["linea"]).all():
        problemas.append("linea no coincide con linea_codigo según config.LINEAS")
    if not (df["es_muebles"] == (df["linea_codigo"] == LINEA_OBJETIVO_CODIGO).astype(int)).all():
        problemas.append("es_muebles no coincide con linea_codigo == 8")

    # Valores
    if (df[OBJETIVO] <= 0).any():
        problemas.append(f"{OBJETIVO} tiene valores menores o iguales a 0 (no admite logaritmo)")
    if not df["covid"].isin([0, 1]).all():
        problemas.append("covid solo puede valer 0 o 1")
    for c in [*PREDICTORES_NUM, *PREDICTORES_REZ1]:
        if df[c].equals(df[OBJETIVO]):
            problemas.append(f"el predictor {c} es idéntico al objetivo (fuga)")

    # Cuadrícula completa: cada fecha × departamento × línea aparece exactamente una vez
    claves = ["fecha", "departamento", "linea_codigo"]
    repetidas = int(df.duplicated(subset=claves).sum())
    if repetidas:
        problemas.append(f"{repetidas} filas repetidas en {claves}")
    esperadas = df["fecha"].nunique() * df["departamento"].nunique() * df["linea_codigo"].nunique()
    if len(df) - repetidas != esperadas:
        unicas = len(df) - repetidas
        problemas.append(f"la cuadrícula está incompleta: {unicas} de {esperadas} filas")

    _lanzar(problemas)
    return df


def _lanzar(problemas: list[str]) -> None:
    if problemas:
        detalle = "\n  - ".join(problemas)
        raise ValueError(f"La base no cumple el contrato:\n  - {detalle}")

"""Agregación ponderada del panel a una línea nacional."""

import pandas as pd

from src.config import LINEA_OBJETIVO, LINEAS, PESOS_DEPTO


def agregar_nacional(
    df: pd.DataFrame, columna: str, linea: str | int = LINEA_OBJETIVO
) -> pd.Series:
    """Suma ``peso_d * columna`` por fecha para la línea solicitada.

    Acepta el nombre de línea o su código EMC, para conservar compatibilidad
    con llamadas previas que pasaban ``linea_codigo`` como tercer argumento.
    """
    if columna not in df:
        raise ValueError(f"No existe la columna {columna!r}")
    if isinstance(linea, int):
        if linea not in LINEAS:
            raise ValueError(f"Código de línea no configurado: {linea}")
        mascara = df["linea_codigo"] == linea
    else:
        if linea not in LINEAS.values():
            raise ValueError(f"Línea no configurada: {linea!r}")
        mascara = df["linea"] == linea
    datos = df.loc[mascara, ["fecha", "departamento", columna]].copy()
    datos["peso"] = datos["departamento"].map(PESOS_DEPTO)
    if datos.empty or datos["peso"].isna().any():
        raise ValueError("La línea debe incluir departamentos con pesos configurados")
    return (datos[columna] * datos["peso"]).groupby(datos["fecha"]).sum().sort_index()

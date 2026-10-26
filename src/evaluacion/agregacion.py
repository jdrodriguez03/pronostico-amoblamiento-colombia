"""Agregación ponderada del panel a una línea nacional."""

import pandas as pd

from src.config import LINEA_OBJETIVO_CODIGO, LINEAS, PESOS_DEPTO


def agregar_nacional(
    df: pd.DataFrame,
    columna: str,
    linea_codigo: str | int = LINEA_OBJETIVO_CODIGO,
    *,
    linea: str | int | None = None,
) -> pd.Series:
    """Suma ``peso_d * columna`` por fecha para la línea solicitada.

    Acepta el nombre de línea o su código EMC. ``linea`` se mantiene como alias
    de compatibilidad para las llamadas por nombre que ya usa el panel de E2.
    """
    if linea is not None:
        linea_codigo = linea
    if columna not in df:
        raise ValueError(f"No existe la columna {columna!r}")
    if isinstance(linea_codigo, int):
        if linea_codigo not in LINEAS:
            raise ValueError(f"Código de línea no configurado: {linea_codigo}")
        mascara = df["linea_codigo"] == linea_codigo
    else:
        if linea_codigo not in LINEAS.values():
            raise ValueError(f"Línea no configurada: {linea_codigo!r}")
        mascara = df["linea"] == linea_codigo
    datos = df.loc[mascara, ["fecha", "departamento", columna]].copy()
    datos["peso"] = datos["departamento"].map(PESOS_DEPTO)
    if datos.empty or datos["peso"].isna().any():
        raise ValueError("La línea debe incluir departamentos con pesos configurados")
    return (datos[columna] * datos["peso"]).groupby(datos["fecha"]).sum().sort_index()

# Dueño: Julián
"""Rezagos realistas según la disponibilidad de cada predictor.

El pronóstico del mes t se emite el primer día hábil del mes t. Cada predictor
entra con el menor rezago k tal que el dato de t-k ya está publicado ese día
(ver docs/disponibilidad_datos.md). Solo se mira al pasado de la misma serie
departamento x línea, así que no hay fuga.
"""

import pandas as pd

PREDICTORES = [
    "icc",
    "inflacion_anual",
    "tasa_consumo",
    "importaciones_muebles_usd_miles",
    "area_vivienda_m2",
]

# Verificado con los calendarios de difusión (docs/disponibilidad_datos.md).
REZAGOS_REALES: dict[str, int] = {
    "icc": 2,
    "inflacion_anual": 2,
    "tasa_consumo": 1,
    "importaciones_muebles_usd_miles": 3,
    "area_vivienda_m2": 3,
}

CLAVES = ["departamento", "linea_codigo"]


def columnas_predictoras(rezagos: dict[str, int] = REZAGOS_REALES) -> list[str]:
    """Nombres de las columnas rezagadas, en el orden de PREDICTORES."""
    return [f"{v}_rez{rezagos[v]}" for v in PREDICTORES if v in rezagos]


def agregar_rezagos_reales(
    df: pd.DataFrame, rezagos: dict[str, int] = REZAGOS_REALES
) -> pd.DataFrame:
    """Agrega {predictor}_rez{k} y descarta las filas sin dato rezagado.

    El valor de t-k se busca por fecha calendario dentro de cada serie
    departamento x línea. Conserva el índice original y no toca el objetivo.
    """
    faltan = [c for c in [*CLAVES, "fecha", *rezagos] if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas: {faltan}")
    if any(k < 1 for k in rezagos.values()):
        raise ValueError("Todo rezago debe ser >= 1")

    out = df.copy()
    fechas = pd.to_datetime(out["fecha"])
    for var, k in rezagos.items():
        indice = pd.MultiIndex.from_arrays(
            [out["departamento"], out["linea_codigo"], fechas]
        )
        serie = pd.Series(out[var].to_numpy(), index=indice)
        serie = serie[~serie.index.duplicated()]
        destino = pd.MultiIndex.from_arrays(
            [out["departamento"], out["linea_codigo"], fechas - pd.DateOffset(months=k)]
        )
        out[f"{var}_rez{k}"] = serie.reindex(destino).to_numpy()

    return out.dropna(subset=columnas_predictoras(rezagos))

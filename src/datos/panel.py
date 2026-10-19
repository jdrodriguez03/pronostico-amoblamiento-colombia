"""Construcción del panel sintético de índices por mes, departamento y línea."""

import pandas as pd

from src.config import (
    DEPARTAMENTOS,
    ESCENARIOS,
    LINEA_OBJETIVO_CODIGO,
    LINEAS,
    N_MESES,
    SEMILLA,
)
from src.datos.emc import factor_regional, leer_departamental, leer_lineas_nacional
from src.datos.simulacion import conservar, simular_eta


def _construir_panel_y_componentes(escenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye panel y tabla separada de auditoría para uso del script."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario inválido: {escenario!r}. Opciones: {list(ESCENARIOS)}")

    nacional = leer_lineas_nacional()
    regional = factor_regional(leer_departamental())
    fechas = pd.DatetimeIndex(sorted(nacional["fecha"].unique()))
    if len(fechas) != N_MESES:
        raise ValueError(f"Se esperaban {N_MESES} meses nacionales, se encontraron {len(fechas)}")

    L = nacional.pivot(index="fecha", columns="linea", values="L").reindex(
        index=fechas, columns=list(LINEAS.values())
    )
    F = regional.pivot(index="fecha", columns="departamento", values="factor_regional").reindex(
        index=fechas, columns=DEPARTAMENTOS
    )
    if L.isna().any().any() or F.isna().any().any():
        raise ValueError("Las series EMC no cubren todas las fechas, líneas y departamentos")

    eta = simular_eta(ESCENARIOS[escenario], semilla=SEMILLA)
    eta = conservar(eta, F.to_numpy())
    indice = L.to_numpy()[:, None, :] * F.to_numpy()[:, :, None] * eta

    filas = []
    componentes = []
    for ti, fecha in enumerate(fechas):
        for di, departamento in enumerate(DEPARTAMENTOS):
            for li, (codigo, linea) in enumerate(LINEAS.items()):
                valor_l = L.iloc[ti, li]
                factor = F.iloc[ti, di]
                filas.append(
                    (fecha, fecha.year, fecha.month, departamento, codigo, linea,
                     int(codigo == LINEA_OBJETIVO_CODIGO), indice[ti, di, li])
                )
                componentes.append(
                    (fecha, departamento, codigo, linea, valor_l, factor, eta[ti, di, li])
                )

    panel = pd.DataFrame(
        filas,
        columns=["fecha", "anio", "mes", "departamento", "linea_codigo", "linea",
                 "es_muebles", "indice_ventas_real"],
    )
    componentes_df = pd.DataFrame(
        componentes,
        columns=["fecha", "departamento", "linea_codigo", "linea", "L", "F", "eta"],
    )
    return panel, componentes_df


def construir_panel(escenario: str) -> pd.DataFrame:
    """Construye el panel para un escenario configurado."""
    panel, _ = _construir_panel_y_componentes(escenario)
    return panel

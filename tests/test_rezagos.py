# Dueño: Julián
import numpy as np
import pandas as pd
import pytest

from src.config import OBJETIVO
from src.datos.cargar import cargar_entrenamiento
from src.features.rezagos import (
    PREDICTORES,
    REZAGOS_REALES,
    agregar_rezagos_reales,
    columnas_predictoras,
)


@pytest.fixture(scope="module")
def base():
    return cargar_entrenamiento(fixture=True)


def test_valor_es_el_de_t_menos_k(base):
    out = agregar_rezagos_reales(base, {"icc": 2})
    fechas = pd.to_datetime(base["fecha"])
    clave = base.set_index(["departamento", "linea_codigo", fechas])["icc"]
    for _, fila in out.sample(20, random_state=42).iterrows():
        ref = pd.to_datetime(fila["fecha"]) - pd.DateOffset(months=2)
        esperado = clave.loc[(fila["departamento"], fila["linea_codigo"], ref)]
        assert fila["icc_rez2"] == pytest.approx(esperado)


def test_k1_reproduce_columnas_rez1(base):
    original = base.copy()
    out = agregar_rezagos_reales(base, {v: 1 for v in PREDICTORES})
    for v in PREDICTORES:
        np.testing.assert_allclose(
            out[f"{v}_rez1"].to_numpy(),
            original.loc[out.index, f"{v}_rez1"].to_numpy(),
            rtol=1e-12,
        )


def test_sin_vacios_y_descarta_primeros_meses(base):
    out = agregar_rezagos_reales(base)
    assert not out[columnas_predictoras()].isna().any().any()
    inicio = pd.to_datetime(base["fecha"]).min()
    k_max = max(REZAGOS_REALES.values())
    assert pd.to_datetime(out["fecha"]).min() == inicio + pd.DateOffset(months=k_max)


def test_objetivo_no_cambia(base):
    out = agregar_rezagos_reales(base)
    pd.testing.assert_series_equal(out[OBJETIVO], base.loc[out.index, OBJETIVO])


def test_columnas_predictoras_nombres():
    assert columnas_predictoras({"icc": 2, "tasa_consumo": 1}) == [
        "icc_rez2",
        "tasa_consumo_rez1",
    ]

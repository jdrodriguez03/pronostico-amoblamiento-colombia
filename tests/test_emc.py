"""Pruebas de lectura y agregación de los anexos EMC originales."""

import numpy as np
import pandas as pd
import pytest

from src.config import DEPARTAMENTOS, FECHA_FIN, FECHA_INICIO, LINEAS, N_MESES, PESOS_DEPTO
from src.datos.emc import (
    ANEXO_DEPARTAMENTAL,
    ANEXO_NACIONAL,
    factor_regional,
    leer_departamental,
    leer_lineas_nacional,
)
from src.simulation import simular_base as legado


@pytest.fixture(scope="module")
def emc():
    nacional = leer_lineas_nacional()
    departamental = leer_departamental()
    factores = factor_regional(departamental)
    return nacional, departamental, factores


def test_lineas_nacionales_tienen_dimensiones_esperadas(emc):
    nacional, _, _ = emc
    assert len(nacional) == 728
    assert nacional["linea"].nunique() == len(LINEAS) == 8
    assert nacional["fecha"].nunique() == N_MESES == 91


def test_indices_departamentales_tienen_dimensiones_esperadas(emc):
    _, departamental, _ = emc
    assert len(departamental) == 637
    assert departamental["departamento"].nunique() == len(DEPARTAMENTOS) == 7
    assert departamental["fecha"].nunique() == N_MESES == 91


def test_series_emc_no_tienen_nulos_en_columnas_requeridas(emc):
    nacional, departamental, factores = emc
    assert not nacional[["fecha", "linea", "L"]].isna().any().any()
    assert not departamental[["fecha", "departamento", "indice_depto"]].isna().any().any()
    assert not factores[["fecha", "departamento", "factor_regional"]].isna().any().any()


def test_factores_regionales_suman_uno_ponderados_por_mes(emc):
    _, _, factores = emc
    pesos = factores["departamento"].map(PESOS_DEPTO)
    suma_ponderada = (factores["factor_regional"] * pesos).groupby(factores["fecha"]).sum()
    np.testing.assert_allclose(suma_ponderada.to_numpy(), 1.0, rtol=0, atol=1e-6)


def test_lectura_departamental_coincide_con_matriz_historica(emc, monkeypatch):
    _, departamental, _ = emc

    libros_abiertos = []
    load_workbook_original = legado.openpyxl.load_workbook

    def load_workbook_registrando(*args, **kwargs):
        libro = load_workbook_original(*args, **kwargs)
        libros_abiertos.append(libro)
        return libro

    monkeypatch.setattr(legado.openpyxl, "load_workbook", load_workbook_registrando)
    try:
        fechas_legado, _, _ = legado.leer_lineas(ANEXO_NACIONAL)
        _, G, _, _ = legado.leer_departamentos(ANEXO_DEPARTAMENTAL, len(fechas_legado))
    finally:
        for libro in libros_abiertos:
            libro.close()

    fechas_esperadas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="MS")
    assert fechas_legado.equals(fechas_esperadas)
    assert tuple(DEPARTAMENTOS) == tuple(legado.DEPTOS)
    assert tuple(departamental["departamento"].drop_duplicates()) == tuple(legado.DEPTOS)
    fechas_nuevas = pd.DatetimeIndex(sorted(departamental["fecha"].unique()))
    assert fechas_nuevas.equals(fechas_legado)

    tabla = departamental.pivot(index="fecha", columns="departamento", values="indice_depto")
    tabla_alineada = tabla.reindex(index=fechas_legado, columns=legado.DEPTOS)
    assert not tabla_alineada.isna().any().any()
    assert tabla_alineada.index.equals(fechas_legado)
    assert tuple(tabla_alineada.columns) == tuple(legado.DEPTOS)
    assert G.shape == (len(fechas_legado), len(legado.DEPTOS))
    np.testing.assert_array_equal(tabla_alineada.to_numpy(), G)

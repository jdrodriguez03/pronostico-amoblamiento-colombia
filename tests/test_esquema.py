# Dueño: Aleja
"""Las tres bases reales cumplen el contrato, y el contrato atrapa los errores típicos."""

import numpy as np
import pandas as pd
import pytest

from src.config import ESCENARIOS, ruta_base
from src.datos.esquema import COLUMNAS_BASE, validar_base


@pytest.fixture(scope="module")
def base():
    return pd.read_csv(ruta_base("central"), parse_dates=["fecha"])


@pytest.mark.parametrize("escenario", list(ESCENARIOS))
def test_bases_reales_cumplen_el_contrato(escenario):
    df = pd.read_csv(ruta_base(escenario), parse_dates=["fecha"])
    validar_base(df)
    assert list(df.columns) == COLUMNAS_BASE
    assert len(df) == 5096
    assert df["area_vivienda_m2_rez1"].isna().sum() == 56


def test_escenarios_solo_cambian_el_objetivo():
    bajo = pd.read_csv(ruta_base("bajo"))
    alto = pd.read_csv(ruta_base("alto"))
    distintas = [c for c in COLUMNAS_BASE if not bajo[c].equals(alto[c])]
    assert distintas == ["indice_ventas_real"]


def test_falta_columna(base):
    with pytest.raises(ValueError, match="faltan columnas"):
        validar_base(base.drop(columns="icc"))


@pytest.mark.parametrize("prohibida", ["factor_regional", "peso_base2019", "L_8"])
def test_columna_de_auditoria(base, prohibida):
    df = base.copy()
    df[prohibida] = 1.0
    with pytest.raises(ValueError, match="fuga"):
        validar_base(df)


def test_predictor_igual_al_objetivo(base):
    df = base.copy()
    df["icc"] = df["indice_ventas_real"]
    with pytest.raises(ValueError, match="idéntico al objetivo"):
        validar_base(df)


def test_vacio_fuera_de_enero_2019(base):
    df = base.copy()
    df.loc[df["fecha"] == "2020-05-01", "tasa_consumo"] = np.nan
    with pytest.raises(ValueError, match="vacíos"):
        validar_base(df)


def test_fila_repetida(base):
    df = pd.concat([base, base.iloc[[100]]], ignore_index=True)
    with pytest.raises(ValueError, match="repetidas"):
        validar_base(df)


def test_cuadricula_incompleta(base):
    with pytest.raises(ValueError, match="incompleta"):
        validar_base(base.iloc[1:])


def test_linea_mal_codificada(base):
    df = base.copy()
    df.loc[0, "linea"] = "Muebles"
    with pytest.raises(ValueError, match="linea_codigo"):
        validar_base(df)

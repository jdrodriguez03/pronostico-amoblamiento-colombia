# Dueño: Aleja
import numpy as np
import pandas as pd
import pytest

from src.config import DEPARTAMENTOS, LINEAS
from src.datos.esquema import COLUMNAS_DATASET, validar_dataset, validar_predictores


def dataset_valido(n_meses=24):
    """Un dataset pequeño que cumple el contrato (2 departamentos x 2 líneas)."""
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2019-01-01", periods=n_meses, freq="MS")
    idx = pd.MultiIndex.from_product(
        [fechas, DEPARTAMENTOS[:2], LINEAS[:2]], names=["fecha", "departamento", "linea"]
    )
    df = idx.to_frame(index=False)
    n = len(df)
    df["mes"] = df["fecha"].dt.month
    df["indice"] = 100 + rng.normal(0, 5, n)
    for c in ["icc", "area_aprobada", "tasa_consumo", "inflacion", "importaciones"]:
        df[c] = rng.normal(0, 1, n)
    df["pandemia"] = 0
    return df[COLUMNAS_DATASET]


def test_dataset_valido_pasa():
    validar_dataset(dataset_valido())


def test_falta_columna():
    with pytest.raises(ValueError, match="faltan columnas"):
        validar_dataset(dataset_valido().drop(columns="icc"))


@pytest.mark.parametrize("prohibida", ["L", "F", "eta"])
def test_columna_prohibida(prohibida):
    df = dataset_valido()
    df[prohibida] = 1.0
    with pytest.raises(ValueError, match="prohibidas"):
        validar_dataset(df)


def test_predictor_igual_al_objetivo():
    df = dataset_valido()
    df["icc"] = df["indice"]
    with pytest.raises(ValueError, match="idéntico al objetivo"):
        validar_dataset(df)


def test_nulos():
    df = dataset_valido()
    df.loc[0, "tasa_consumo"] = np.nan
    with pytest.raises(ValueError, match="nulos"):
        validar_dataset(df)


def test_fila_repetida():
    df = dataset_valido()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repetidas"):
        validar_dataset(df)


def test_cuadricula_incompleta():
    with pytest.raises(ValueError, match="incompleta"):
        validar_dataset(dataset_valido().iloc[1:])


def test_departamento_desconocido():
    df = dataset_valido()
    df.loc[df["departamento"] == DEPARTAMENTOS[0], "departamento"] = "Narnia"
    with pytest.raises(ValueError, match="departamentos"):
        validar_dataset(df)


def test_predictores_valida():
    fechas = pd.date_range("2019-01-01", periods=3, freq="MS")
    df = pd.MultiIndex.from_product([fechas, DEPARTAMENTOS], names=["fecha", "departamento"])
    df = df.to_frame(index=False)
    for c in ["icc", "area_aprobada", "tasa_consumo", "inflacion", "importaciones"]:
        df[c] = 1.0
    df["pandemia"] = 0
    validar_predictores(df)
    df.loc[0, "pandemia"] = 2
    with pytest.raises(ValueError, match="pandemia"):
        validar_predictores(df)

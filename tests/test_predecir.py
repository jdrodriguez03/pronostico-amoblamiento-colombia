# Dueño: Aleja
"""La lógica de la app predice un número positivo con las mismas columnas del entrenamiento."""

import numpy as np
import pandas as pd

from src.config import FECHA_FIN
from src.datos.cargar import cargar_entrenamiento
from src.despliegue.predecir import (
    NACIONAL,
    _agregar_mes_siguiente,
    drift_predictores,
    entrenar_para_app,
    filas_a_predecir,
    fuera_de_rango,
    predecir,
    rangos_entrenamiento,
    semaforo_drift,
    tabla_entradas,
)
from src.features.rezagos import REZAGOS_REALES, agregar_rezagos_reales, columnas_predictoras
from src.modelos.pipeline import preparar_datos


def _insumos_fixture() -> pd.DataFrame:
    """Lo mismo que cargar_insumos_app, pero con el fixture (sin leer la base completa)."""
    df = pd.read_csv("data/fixtures/base_mini.csv", parse_dates=["fecha"])
    return agregar_rezagos_reales(_agregar_mes_siguiente(df), REZAGOS_REALES)


def test_prediccion_es_numero_positivo():
    modelo, _ = entrenar_para_app(fixture=True)
    df = _insumos_fixture()
    filas = filas_a_predecir(df, "Antioquia", 8, "2025-08-01")
    pred = predecir(modelo, filas)
    assert np.isfinite(pred) and pred > 0


def test_nacional_y_mes_siguiente():
    modelo, _ = entrenar_para_app(fixture=True)
    df = _insumos_fixture()
    siguiente = pd.Timestamp(FECHA_FIN) + pd.DateOffset(months=1)
    filas = filas_a_predecir(df, NACIONAL, 8, siguiente)
    assert filas[columnas_predictoras()].notna().all().all()
    assert predecir(modelo, filas) > 0


def test_columnas_de_entrada_coinciden_con_entrenamiento():
    _, predictores = preparar_datos(cargar_entrenamiento(fixture=True), "reales")
    filas = filas_a_predecir(_insumos_fixture(), "Bogota", 4, "2024-03-01")
    assert tabla_entradas(filas)["columna"].tolist() == predictores


def test_valor_extremo_se_marca_fuera_de_rango():
    df = _insumos_fixture()
    tabla = tabla_entradas(filas_a_predecir(df, "Bogota", 8, "2024-03-01"))
    rangos = rangos_entrenamiento(df)
    assert fuera_de_rango(tabla, rangos) == []
    tabla.loc[0, "valor"] = rangos.loc[tabla.loc[0, "columna"], "max"] + 1000
    assert fuera_de_rango(tabla, rangos) == [tabla.loc[0, "predictor"]]


def test_drift_una_fila_por_predictor_y_semaforo():
    tabla = drift_predictores(_insumos_fixture())
    assert len(tabla) == len(REZAGOS_REALES)
    assert tabla["p-valor"].between(0, 1).all()
    assert semaforo_drift(tabla).split(":")[0] in {"VERDE", "AMARILLO", "ROJO"}

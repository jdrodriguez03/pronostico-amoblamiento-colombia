# Dueño: Juanes
"""Prueba de punta a punta de scripts/entrenar.py en modo --rapido (base pequeña).

Corre en el CI en pocos segundos y comprueba la forma de las tres tablas, que no se lee
la prueba y que las búsquedas quedan guardadas para las curvas de Julián.
"""

import importlib.util
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.config import CV_FOLDS, ESCENARIOS, INICIO_PRUEBA
from src.modelos.candidatos import CANDIDATOS

RUTA_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "entrenar.py"
N_MODELOS = len(CANDIDATOS) + 2  # 5 candidatos + 2 líneas base


def _cargar_script():
    """Importa scripts/entrenar.py como módulo (scripts/ no es un paquete)."""
    spec = importlib.util.spec_from_file_location("entrenar", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    """Una corrida --rapido de los 3 escenarios en una carpeta temporal."""
    entrenar = _cargar_script()
    carpeta = tmp_path_factory.mktemp("modelos")
    mp = pytest.MonkeyPatch()
    mp.setattr(entrenar, "RUTA_MODELOS", carpeta)
    try:
        cv = entrenar.main(["--rapido", "--escenario", "todos"])
    finally:
        mp.undo()
    return entrenar, carpeta / "rapido", cv


def test_cv_resultados_tiene_7_modelos_3_escenarios_5_folds(corrida):
    entrenar, carpeta, cv = corrida
    assert len(cv) == N_MODELOS * len(ESCENARIOS) * CV_FOLDS
    conteo = cv.groupby(["escenario", "modelo"])["fold"].nunique()
    assert (conteo == CV_FOLDS).all()
    leido = pd.read_csv(carpeta / "cv_resultados.csv")
    assert list(leido.columns) == entrenar.COLUMNAS
    assert leido[["mape", "mad", "msd", "rmse", "mape_entrena", "mape_ajuste"]].notna().all().all()


def test_nunca_valida_con_meses_de_la_prueba(corrida):
    _, _, cv = corrida
    assert (pd.to_datetime(cv["valida_hasta"]) < pd.Timestamp(INICIO_PRUEBA)).all()


def test_rmse_mayor_o_igual_que_mad(corrida):
    _, _, cv = corrida
    assert (cv["rmse"] >= cv["mad"] - 1e-9).all()


def test_tablas_de_comparacion(corrida):
    _, carpeta, _ = corrida
    comparacion = pd.read_csv(carpeta / "comparacion_cv.csv")
    assert len(comparacion) == N_MODELOS * len(ESCENARIOS)
    pareada = pd.read_csv(carpeta / "comparacion_pareada.csv")
    # 2 referencias; cada una se compara con los otros 6 modelos en cada escenario
    assert len(pareada) == 2 * (N_MODELOS - 1) * len(ESCENARIOS)


def test_busquedas_guardadas_con_resultados_de_entrenamiento(corrida):
    _, carpeta, _ = corrida
    busqueda = joblib.load(carpeta / "central_ridge.joblib")
    assert "split0_train_score" in busqueda.cv_results_
    assert (carpeta / "central_ingenuo_estacional.joblib").exists()


def test_rejilla_rapida_y_texto_de_hiperparametros():
    entrenar = _cargar_script()
    assert entrenar.rejilla_rapida({"a": [1, 2, 3], "b": ["x"]}) == {"a": [1, 3], "b": ["x"]}
    texto = entrenar.texto_hiperparametros(
        {"regressor__modelo__n_neighbors": 3, "regressor__modelo__weights": "uniform"})
    assert texto == "n_neighbors=3; weights=uniform"
    assert entrenar.texto_hiperparametros({}) == "sin hiperparámetros"

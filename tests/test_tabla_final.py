# Dueño: Juanes
"""Prueba de scripts/tabla_final.py con tablas INVENTADAS (no lee la prueba real)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUTA_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tabla_final.py"
MODELOS = ["ridge", "knn", "mlr_stepwise", "media", "ingenuo_estacional"]


def _cargar_script():
    spec = importlib.util.spec_from_file_location("tabla_final", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def tablas():
    rng = np.random.default_rng(1)
    fechas = pd.date_range("2030-01-01", periods=12, freq="MS")
    y = 150 + rng.normal(0, 10, 12)
    sesgo = {"ridge": 0.75, "knn": 0.85, "mlr_stepwise": 0.7, "media": 0.72,
             "ingenuo_estacional": 0.88}
    pred, prueba, cv = [], [], []
    for escenario in ("bajo", "central", "alto"):
        for m in MODELOS:
            yhat = y * sesgo[m] * (1 + rng.normal(0, 0.03, 12))
            pred.append(pd.DataFrame({"escenario": escenario, "fecha": fechas,
                                      "ambito": "Nacional", "linea_codigo": 8, "modelo": m,
                                      "y_real": y, "y_pred": yhat}))
            prueba.append({"escenario": escenario, "modelo": m,
                           "mape": float(np.mean(np.abs(y - yhat) / y) * 100),
                           "hiperparametros": "x"})
            cv.append({"escenario": escenario, "modelo": m, "mape_media": 9.0,
                       "mape_desv": 2.0})
    return pd.DataFrame(prueba), pd.concat(pred, ignore_index=True), pd.DataFrame(cv)


def test_tabla_final_formato_e_intervalos(tablas):
    script = _cargar_script()
    tabla = script.construir_tabla_final(*tablas).set_index("modelo")
    assert list(tabla.index) == MODELOS
    assert (tabla["mape_prueba_ic_inf"] <= tabla["mape_prueba"]).all()
    assert (tabla["mape_prueba"] <= tabla["mape_prueba_ic_sup"]).all()
    # El ingenuo contra sí mismo: diferencia 0 exacta
    assert tabla.loc["ingenuo_estacional", "dif_vs_ingenuo"] == 0
    # La diferencia Ridge − KNN solo va en sus dos filas, y es la misma en las dos
    assert tabla.loc["ridge", "dif_ridge_menos_knn"] == tabla.loc["knn", "dif_ridge_menos_knn"]
    assert pd.isna(tabla.loc["media", "dif_ridge_menos_knn"])
    assert (tabla["n_meses"] == 12).all() and (tabla["largo_bloque"] == 3).all()


def test_tabla_final_falla_si_no_cuadra_con_prueba_csv(tablas):
    script = _cargar_script()
    prueba, pred, cv = tablas
    mala = prueba.copy()
    mala.loc[(mala["escenario"] == "central") & (mala["modelo"] == "knn"), "mape"] += 1
    with pytest.raises(ValueError, match="no coincide con prueba.csv"):
        script.construir_tabla_final(mala, pred, cv)

# Dueño: Juanes
"""Pruebas del modelo final y de las tablas del martes 10 que salen de la validación cruzada.

Con la base pequeña: aquí no se lee la prueba.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import OBJETIVO
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.comparacion import MODELOS_COSTO_REZAGOS, tabla_costo_rezagos
from src.evaluacion.particion import CVTemporalPorMes
from src.evaluacion.validar import buscar_hiperparametros, evaluar_cv, predicciones_oof
from src.features.rezagos import REZAGOS_REALES, columnas_predictoras
from src.modelos.candidatos import CANDIDATOS
from src.modelos.final import ESPECIFICACION_FINAL, HIPERPARAMETROS, construir_modelo_final
from src.modelos.pipeline import prefijo_parametros, preparar_datos


@pytest.fixture(scope="module")
def mini():
    datos, _ = preparar_datos(cargar_entrenamiento(fixture=True), "reales")
    return datos


def test_especificacion_tiene_lo_que_pide_el_contrato():
    for clave in ("modelo", "hiperparametros", "rezagos", "escenario"):
        assert clave in ESPECIFICACION_FINAL
    assert ESPECIFICACION_FINAL["modelo"] in {"ridge", "knn"}
    assert ESPECIFICACION_FINAL["rezagos"] == REZAGOS_REALES
    assert ESPECIFICACION_FINAL["escenario"] == "central"


def test_hiperparametros_estan_en_la_rejilla_congelada():
    """El valor lo elige la validación, pero tiene que ser un punto de la rejilla."""
    rejilla_ridge = CANDIDATOS["ridge"][2]["alpha"]
    assert np.isclose(rejilla_ridge, HIPERPARAMETROS["ridge"]["alpha"]).any()
    rejilla_knn = CANDIDATOS["knn"][2]
    assert HIPERPARAMETROS["knn"]["n_neighbors"] in rejilla_knn["n_neighbors"]
    assert HIPERPARAMETROS["knn"]["weights"] in rejilla_knn["weights"]


@pytest.mark.parametrize("nombre", ["ridge", "knn"])
def test_construir_modelo_final_sin_ajustar_y_con_valores_fijos(nombre):
    modelo = construir_modelo_final(nombre)
    params = modelo.get_params()
    for clave, valor in HIPERPARAMETROS[nombre].items():
        assert params[f"{prefijo_parametros()}{clave}"] == valor
    assert not hasattr(modelo, "regressor_")  # sin ajustar


def test_construir_modelo_final_rechaza_otro_modelo():
    with pytest.raises(ValueError, match="ridge"):
        construir_modelo_final("lasso")


def test_modelo_final_es_el_mismo_que_eligio_gridsearch(mini):
    """Con un alpha fijo, el MAPE por fold es el de GridSearchCV con una rejilla de un punto."""
    alpha = HIPERPARAMETROS["ridge"]["alpha"]
    busqueda = buscar_hiperparametros("ridge", mini, columnas_predictoras(),
                                      rejilla={f"{prefijo_parametros()}alpha": [alpha]})
    folds = evaluar_cv(construir_modelo_final("ridge"), mini, CVTemporalPorMes())
    assert -busqueda.best_score_ == pytest.approx(folds["mape"].mean())


def test_predicciones_oof_son_los_meses_de_validacion(mini):
    cv = CVTemporalPorMes()
    oof = predicciones_oof(construir_modelo_final("ridge"), mini, cv)
    assert list(oof.columns) == ["fold", "fecha", "departamento", "linea_codigo",
                                 "y_real", "y_pred"]
    ventanas = cv.ventanas(mini)
    assert oof["fold"].nunique() == len(ventanas)
    # 30 meses distintos, sin repetir una fila, todos dentro de la validación
    assert pd.to_datetime(oof["fecha"]).nunique() == 30
    assert not oof.duplicated(["fecha", "departamento", "linea_codigo"]).any()
    assert oof["y_real"].to_numpy() == pytest.approx(
        mini.set_index(["fecha", "departamento", "linea_codigo"])
        .loc[pd.MultiIndex.from_frame(oof[["fecha", "departamento", "linea_codigo"]]), OBJETIVO]
        .to_numpy())


def test_tabla_costo_rezagos_a_mano():
    filas = []
    for i, modelo in enumerate(MODELOS_COSTO_REZAGOS):
        filas.append({"escenario": "central", "modelo": modelo, "mape_media": 10.0 + i,
                      "mape_desv": 1.0, "hiperparametros": "x"})
    rez1 = pd.DataFrame(filas)
    reales = rez1.assign(mape_media=rez1["mape_media"] + 0.5)
    tabla = tabla_costo_rezagos(rez1, reales)
    assert list(tabla["modelo"]) == MODELOS_COSTO_REZAGOS
    assert tabla["diferencia"].to_numpy() == pytest.approx(0.5)
    assert tabla.loc[0, "diferencia_pct"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="faltan"):
        tabla_costo_rezagos(rez1[rez1["modelo"] != "knn"], reales)

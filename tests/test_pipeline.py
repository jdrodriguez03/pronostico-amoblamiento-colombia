# Dueño: Juanes
"""Pruebas del preprocesador y del Pipeline, incluida la prueba de fuga del día 4."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from src.config import OBJETIVO, predictores_modelo
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.particion import CVTemporalPorMes
from src.modelos.pipeline import construir_pipeline, prefijo_parametros
from src.modelos.preprocesamiento import columnas_log_por_defecto, crear_preprocesador


@pytest.fixture(scope="module")
def mini():
    return cargar_entrenamiento(fixture=True)


@pytest.fixture(scope="module")
def fold1(mini):
    entrena, valida = next(iter(CVTemporalPorMes().split(mini)))
    return mini.iloc[entrena], mini.iloc[valida]


def test_pipeline_entrena_y_predice_en_escala_original(fold1):
    ent, val = fold1
    modelo = construir_pipeline(LinearRegression()).fit(ent, ent[OBJETIVO])
    pred = modelo.predict(val)
    assert pred.shape == (len(val),)
    assert np.isfinite(pred).all()
    # Escala del índice (2019 = 100), no del logaritmo (≈ 4,6)
    assert pred.mean() > 30


def test_prueba_de_fuga_scaler_solo_ve_el_fold(mini, fold1):
    """La media del StandardScaler ajustado en el fold 1 es la de esos meses, no la de toda
    la base. Si alguien escalara antes de partir, esta prueba fallaría."""
    ent, _ = fold1
    modelo = construir_pipeline(LinearRegression()).fit(ent, ent[OBJETIVO])
    prep = modelo.regressor_.named_steps["prep"]

    escalador = prep.named_transformers_["num"]
    columnas = list(escalador.feature_names_in_)
    np.testing.assert_allclose(escalador.mean_, ent[columnas].mean().to_numpy())
    assert not np.allclose(escalador.mean_, mini[columnas].mean().to_numpy())

    # Lo mismo para el área de vivienda, que se escala después del logaritmo
    vivienda = prep.named_transformers_["log"].named_steps["escalar"]
    col_log = columnas_log_por_defecto()
    np.testing.assert_allclose(vivienda.mean_, np.log(ent[col_log]).mean().to_numpy())


def test_columnas_que_entran_y_salen(fold1):
    ent, _ = fold1
    prep = crear_preprocesador().fit(ent)
    nombres = list(prep.get_feature_names_out())
    # 11 meses + 1 departamento + 1 línea (drop="first") + 5 numéricos + covid
    assert len(nombres) == 11 + 1 + 1 + 5 + 1
    texto = " ".join(nombres)
    assert OBJETIVO not in texto  # el objetivo nunca es predictor
    assert "fecha" not in texto
    assert "icc_rez1" in texto
    assert not any(n.endswith("__icc") for n in nombres)  # solo predictores rezagados
    assert set(predictores_modelo()) <= {n.split("__", 1)[1] for n in nombres}


def test_categoria_no_vista_no_rompe(fold1):
    ent, val = fold1
    modelo = construir_pipeline(LinearRegression()).fit(ent, ent[OBJETIVO])
    raro = val.copy()
    raro["departamento"] = "Atlantico"  # no está en el fixture
    # scikit-learn avisa y la codifica como la categoría de referencia: no falla
    with pytest.warns(UserWarning, match="unknown categories"):
        pred = modelo.predict(raro)
    assert np.isfinite(pred).all()


def test_sin_log_objetivo(fold1):
    ent, val = fold1
    pipe = construir_pipeline(LinearRegression(), log_objetivo=False).fit(ent, ent[OBJETIVO])
    assert list(pipe.named_steps) == ["prep", "sel", "modelo"]
    assert np.isfinite(pipe.predict(val)).all()


def test_funciona_con_gridsearch_y_cv_por_mes(mini):
    """Así lo usará scripts/entrenar.py: la rejilla con el prefijo correcto y el CV por mes."""
    p = prefijo_parametros()
    for modelo, rejilla in [
        (Ridge(), {f"{p}alpha": [0.1, 10.0]}),
        (KNeighborsRegressor(), {f"{p}n_neighbors": [3, 5]}),
    ]:
        busqueda = GridSearchCV(construir_pipeline(modelo), rejilla, cv=CVTemporalPorMes(),
                                scoring="neg_mean_absolute_percentage_error")
        busqueda.fit(mini, mini[OBJETIVO])
        assert len(busqueda.cv_results_["mean_test_score"]) == 2


def test_se_puede_clonar():
    # GridSearchCV clona el pipeline en cada fold: debe salir "en blanco"
    copia = clone(construir_pipeline(Ridge(alpha=3.0)))
    assert copia.get_params()["regressor__modelo__alpha"] == 3.0
    assert not hasattr(copia, "regressor_")


def test_columnas_log_invalidas():
    with pytest.raises(ValueError):
        crear_preprocesador(columnas_log=["icc"])  # predictor del mismo mes: no está en el modelo

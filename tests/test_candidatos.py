# Dueño: Julián
"""Los 5 candidatos se construyen y entrenan.

Mientras se fusiona el Pipeline de Juanes, se prueba con un Pipeline mínimo que imita
su estructura (prep → sel → modelo, con nombres de columna "cat__", "num__", "bin__").
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import OBJETIVO, PREDICTORES_BIN, PREDICTORES_CAT, predictores_modelo
from src.datos.cargar import cargar_entrenamiento
from src.modelos.candidatos import CANDIDATOS, construir_candidato, rejilla_pipeline


@pytest.fixture(scope="module")
def base_pequena():
    df = cargar_entrenamiento("central")
    return df[df["departamento"].isin(["Antioquia", "Bogota"])].reset_index(drop=True)


def pipeline_minimo(modelo, selector):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False), PREDICTORES_CAT),
        ("num", StandardScaler(), predictores_modelo()),
        ("bin", "passthrough", PREDICTORES_BIN),
    ])
    pipe = Pipeline([("prep", prep), ("sel", selector or "passthrough"), ("modelo", modelo)])
    return pipe.set_output(transform="pandas")


def test_estan_los_cinco():
    assert set(CANDIDATOS) == {"mlr_stepwise", "mlr_completa", "ridge", "lasso", "knn"}


@pytest.mark.parametrize("nombre", list(CANDIDATOS))
def test_cada_candidato_entrena_y_predice(base_pequena, nombre):
    modelo, selector, rejilla = CANDIDATOS[nombre]
    pipe = pipeline_minimo(clone(modelo), clone(selector) if selector else None)
    # Primer valor de cada hiperparámetro de la rejilla
    pipe.set_params(**{f"modelo__{k}": list(v)[0] for k, v in rejilla.items()})
    y = np.log(base_pequena[OBJETIVO])
    pipe.fit(base_pequena, y)
    pred = pipe.predict(base_pequena)
    assert pred.shape == (len(base_pequena),)
    assert np.isfinite(pred).all()


def test_rejilla_lleva_el_prefijo_del_pipeline():
    assert set(rejilla_pipeline("ridge", log_objetivo=True)) == {"regressor__modelo__alpha"}
    assert set(rejilla_pipeline("knn", log_objetivo=False)) == {
        "modelo__n_neighbors",
        "modelo__weights",
    }
    assert rejilla_pipeline("mlr_completa") == {}


def test_construir_candidato_con_el_pipeline_de_juanes(base_pequena):
    try:
        pipe, rejilla = construir_candidato("mlr_stepwise")
    except NotImplementedError:
        pytest.skip("construir_pipeline de Juanes aún no está en esta rama")
    pipe.fit(base_pequena, base_pequena[OBJETIVO])
    sel = pipe.regressor_.named_steps["sel"]
    nombres = sel.get_feature_names_out()
    # Todas las dummies forzadas siguen ahí
    n_dummies = sum(base_pequena[c].nunique() - 1 for c in PREDICTORES_CAT)
    assert sum(c.startswith("cat__") for c in nombres) == n_dummies
    assert rejilla == {}


def test_knn_usa_one_hot_sin_drop():
    """KNN pide el one-hot sin drop; los modelos lineales conservan drop="first"."""
    pipe_knn, _ = construir_candidato("knn")
    prep_knn = pipe_knn.regressor.named_steps["prep"]
    assert {n: t for n, t, _ in prep_knn.transformers}["cat"].drop is None

    pipe_ridge, _ = construir_candidato("ridge")
    prep_ridge = pipe_ridge.regressor.named_steps["prep"]
    assert {n: t for n, t, _ in prep_ridge.transformers}["cat"].drop == "first"


def test_rejilla_de_ridge_pasa_del_viejo_borde():
    """El tope de alpha supera 1000, donde caía el mínimo con la rejilla anterior."""
    alphas = rejilla_pipeline("ridge")["regressor__modelo__alpha"]
    assert min(alphas) == pytest.approx(1e-3)
    assert max(alphas) == pytest.approx(1e5)

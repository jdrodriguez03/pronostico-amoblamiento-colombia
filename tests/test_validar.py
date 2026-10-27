# Dueño: Juanes
"""Pruebas del scorer del MAPE nacional y de evaluar_cv sobre la base pequeña.

Dependen de agregar_nacional (Aleja). Mientras no esté en main, se saltan solas: así el
CI sigue en verde y, apenas se fusione, empiezan a correr sin tocar nada.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from src.config import LINEA_OBJETIVO_CODIGO, OBJETIVO, PESOS_DEPTO
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.agregacion import agregar_nacional
from src.evaluacion.metricas import mape
from src.evaluacion.particion import CVTemporalPorMes
from src.evaluacion.validar import evaluar_cv, mape_ajuste, scorer_mape_nacional
from src.modelos.lineas_base import IngenuoEstacional, PredictorMedia
from src.modelos.pipeline import construir_pipeline, prefijo_parametros


@pytest.fixture(scope="module")
def mini():
    df = cargar_entrenamiento(fixture=True)
    try:
        agregar_nacional(df, OBJETIVO, LINEA_OBJETIVO_CODIGO)
    except NotImplementedError:
        pytest.skip("agregar_nacional de Aleja aún no está en esta rama")
    return df


class _Constante:
    """'Modelo' de juguete que predice siempre el observado por un factor fijo."""

    def __init__(self, y, factor):
        self.y, self.factor = np.asarray(y, dtype=float), factor

    def predict(self, X):
        return self.y * self.factor


def test_scorer_a_mano(mini):
    """Si todas las filas se pasan 10 %, la serie nacional también: MAPE = 10 %.

    Como la agregación es un promedio ponderado (lineal), multiplicar cada fila por 1,1
    multiplica la serie nacional por 1,1: el error porcentual de cada mes es 10 %."""
    y = mini[OBJETIVO]
    assert scorer_mape_nacional(_Constante(y, 1.10), mini, y) == pytest.approx(-10.0)
    assert scorer_mape_nacional(_Constante(y, 1.00), mini, y) == pytest.approx(0.0)


def test_scorer_es_negativo_y_usa_la_serie_nacional(mini):
    """El scorer coincide con el MAPE de la serie agregada a mano con los pesos."""
    y = mini[OBJETIVO].to_numpy()
    rng = np.random.default_rng(0)
    yhat = y * rng.uniform(0.8, 1.2, size=len(y))

    muebles = mini.assign(yhat=yhat, y=y)
    muebles = muebles[muebles["linea_codigo"] == LINEA_OBJETIVO_CODIGO]
    muebles = muebles.assign(w=muebles["departamento"].map(PESOS_DEPTO))
    obs = (muebles["y"] * muebles["w"]).groupby(muebles["fecha"]).sum()
    pred = (muebles["yhat"] * muebles["w"]).groupby(muebles["fecha"]).sum()

    class _Fijo:
        def predict(self, X):
            return yhat

    score = scorer_mape_nacional(_Fijo(), mini, y)
    assert score < 0  # scikit-learn maximiza: el score es −MAPE
    assert score == pytest.approx(-mape(obs, pred))


def test_evaluar_cv_da_5_folds_con_el_ingenuo(mini):
    """El 'Listo cuando' del plan: una tabla de 5 folds para el ingenuo estacional."""
    tabla = evaluar_cv(IngenuoEstacional(), mini, CVTemporalPorMes())
    assert list(tabla["fold"]) == [1, 2, 3, 4, 5]
    columnas = ["mape", "mad", "msd", "rmse", "mape_panel", "mape_entrena"]
    assert set(columnas) <= set(tabla.columns)
    assert np.isfinite(tabla[columnas].to_numpy()).all()
    assert (tabla["rmse"] >= tabla["mad"] - 1e-9).all()
    np.testing.assert_allclose(tabla["rmse"] ** 2, tabla["msd"])
    # Las ventanas son las de CVTemporalPorMes, en orden y sin solaparse
    assert (tabla["valida_desde"].iloc[1:].to_numpy() > tabla["valida_hasta"].iloc[:-1]
            .to_numpy()).all()


def test_evaluar_cv_con_la_media_y_un_pipeline(mini):
    for modelo in [PredictorMedia(), construir_pipeline(Ridge(alpha=1.0))]:
        tabla = evaluar_cv(modelo, mini, CVTemporalPorMes())
        assert len(tabla) == 5
        assert (tabla["mape"] > 0).all()


def test_evaluar_cv_coincide_con_gridsearch(mini):
    """El MAPE de evaluar_cv en cada fold es el mismo que GridSearchCV calcula con el
    scorer (con signo cambiado). Así las líneas base y los modelos son comparables."""
    pipe = construir_pipeline(Ridge(alpha=1.0))
    busqueda = GridSearchCV(pipe, {f"{prefijo_parametros()}alpha": [1.0]},
                            cv=CVTemporalPorMes(), scoring=scorer_mape_nacional,
                            return_train_score=True).fit(mini, mini[OBJETIVO])
    tabla = evaluar_cv(pipe, mini, CVTemporalPorMes())
    for k in range(5):
        assert -busqueda.cv_results_[f"split{k}_test_score"][0] == pytest.approx(
            tabla["mape"].iloc[k])
        assert -busqueda.cv_results_[f"split{k}_train_score"][0] == pytest.approx(
            tabla["mape_entrena"].iloc[k])


def test_scorer_falla_claro_sin_columnas_de_serie(mini):
    y = mini[OBJETIVO]
    with pytest.raises(ValueError, match="le faltan"):
        scorer_mape_nacional(_Constante(y, 1.0), mini.drop(columns="departamento"), y)


def test_ingenuo_mape_entrena_usa_solo_meses_con_historia(mini):
    """En entrenamiento el ingenuo no puede predecir los 12 primeros meses: evaluar_cv
    los excluye en vez de fallar."""
    tabla = evaluar_cv(IngenuoEstacional(), mini, CVTemporalPorMes())
    assert tabla["mape_entrena"].notna().all()
    assert isinstance(tabla["valida_desde"].iloc[0], pd.Timestamp)


def test_mape_ajuste_es_el_scorer_sobre_los_meses_de_ajuste(mini):
    """mape_ajuste = MAPE nacional del modelo ajustado con todos los meses, sobre esos meses."""
    y = mini[OBJETIVO]
    modelo = PredictorMedia().fit(mini, y)
    assert mape_ajuste(modelo, mini) == pytest.approx(-scorer_mape_nacional(modelo, mini, y))


def test_mape_ajuste_del_ingenuo_empieza_en_el_mes_13(mini):
    """El ingenuo no puede predecir los 12 primeros meses: se miden desde el 13 y no falla."""
    modelo = IngenuoEstacional().fit(mini, mini[OBJETIVO])
    fechas = pd.to_datetime(mini["fecha"])
    desde_13 = mini[fechas >= fechas.min() + pd.DateOffset(months=12)]
    esperado = -scorer_mape_nacional(modelo, desde_13, desde_13[OBJETIVO])
    assert mape_ajuste(modelo, mini) == pytest.approx(esperado)

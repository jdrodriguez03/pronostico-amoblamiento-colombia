# Dueño: Juanes
"""Pruebas del bootstrap por bloques y del rango de error (entrega final, lunes 9).

Las tres que pide el documento de la entrega: misma semilla → mismo resultado; los bloques
contienen meses consecutivos; el IC contiene la estimación puntual. Más las de guarda: que
nunca acepte filas del panel (meses repetidos) y el formato de rango_error.csv.
Todo con series inventadas: aquí no se lee ningún dato de la prueba.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import DEPARTAMENTOS, LINEA_OBJETIVO_CODIGO
from src.evaluacion.incertidumbre import (
    COLUMNAS_RANGO,
    NACIONAL,
    bootstrap_bloques,
    bootstrap_diferencia,
    indices_bloques,
    rango_error,
)
from src.evaluacion.metricas import mape


@pytest.fixture
def serie():
    """12 meses inventados (como la prueba): observado y dos predicciones."""
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2030-01-01", periods=12, freq="MS")
    y = pd.Series(100 + np.arange(12) * 2.0, index=fechas)
    a = y * (1 + rng.normal(0, 0.05, 12))
    b = y * (1 + rng.normal(0.02, 0.06, 12))
    return y, pd.Series(a, index=fechas), pd.Series(b, index=fechas)


# ---------------------------------------------------------------------------
# indices_bloques
# ---------------------------------------------------------------------------

def test_bloques_son_meses_consecutivos():
    idx = indices_bloques(12, largo_bloque=3, B=500, semilla=1)
    assert idx.shape == (500, 12)
    assert idx.min() >= 0 and idx.max() <= 11
    bloques = idx.reshape(500, 4, 3)          # 4 bloques de 3 meses por réplica
    # Dentro de cada bloque: posición, posición + 1, posición + 2
    assert (np.diff(bloques, axis=2) == 1).all()


def test_n_no_multiplo_del_bloque_se_recorta():
    idx = indices_bloques(10, largo_bloque=3, B=50)
    assert idx.shape == (50, 10)              # se sortean 4 bloques (12) y se recorta a 10
    assert (np.diff(idx[:, :9].reshape(50, 3, 3), axis=2) == 1).all()


def test_largo_de_bloque_invalido():
    with pytest.raises(ValueError, match="largo_bloque"):
        indices_bloques(12, largo_bloque=13)


# ---------------------------------------------------------------------------
# bootstrap_bloques y bootstrap_diferencia
# ---------------------------------------------------------------------------

def test_misma_semilla_mismo_resultado(serie):
    y, a, _ = serie
    assert bootstrap_bloques(y, a, semilla=7) == bootstrap_bloques(y, a, semilla=7)
    assert bootstrap_bloques(y, a, semilla=7) != bootstrap_bloques(y, a, semilla=8)


def test_ic_contiene_la_estimacion_puntual(serie):
    y, a, _ = serie
    r = bootstrap_bloques(y, a)
    assert r["mape"] == pytest.approx(mape(y, a))
    assert r["ic_inf"] <= r["mape"] <= r["ic_sup"]
    assert r["ic_inf"] < r["ic_sup"]
    assert r["n_meses"] == 12 and r["largo_bloque"] == 3 and r["B"] == 2000


def test_prediccion_perfecta_da_intervalo_cero(serie):
    y, _, _ = serie
    r = bootstrap_bloques(y, y)
    assert r["mape"] == r["ic_inf"] == r["ic_sup"] == 0


def test_nunca_acepta_filas_del_panel(serie):
    """Con meses repetidos (las filas de un panel) debe fallar: el bootstrap es nacional."""
    y, a, _ = serie
    panel_y = pd.concat([y, y])
    panel_a = pd.concat([a, a])
    with pytest.raises(ValueError, match="meses repetidos"):
        bootstrap_bloques(panel_y, panel_a)


def test_ordena_por_fecha(serie):
    """El resultado no depende del orden en que lleguen las filas: se ordena por fecha."""
    y, a, _ = serie
    orden = [5, 0, 11, 3, 1, 2, 4, 6, 10, 7, 8, 9]
    assert bootstrap_bloques(y.iloc[orden], a.iloc[orden]) == bootstrap_bloques(y, a)


def test_diferencia_es_pareada_y_consistente(serie):
    y, a, b = serie
    d = bootstrap_diferencia(y, a, b)
    assert d["dif"] == pytest.approx(mape(y, a) - mape(y, b))
    assert d["ic_inf"] <= d["dif"] <= d["ic_sup"]
    assert 0 <= d["prop_a_mejor"] <= 1
    # Un modelo contra sí mismo: diferencia exactamente 0 en todas las réplicas
    cero = bootstrap_diferencia(y, a, a)
    assert cero["dif"] == cero["ic_inf"] == cero["ic_sup"] == 0
    # Antisimetría: MAPE(b) − MAPE(a) es el espejo, porque se usan las mismas réplicas
    espejo = bootstrap_diferencia(y, b, a)
    assert espejo["dif"] == pytest.approx(-d["dif"])
    assert espejo["ic_inf"] == pytest.approx(-d["ic_sup"])


# ---------------------------------------------------------------------------
# rango_error
# ---------------------------------------------------------------------------

def _oof_inventado() -> pd.DataFrame:
    """30 meses (como los de validación) × 7 departamentos × 2 líneas, error conocido."""
    rng = np.random.default_rng(3)
    fechas = pd.date_range("2030-01-01", periods=30, freq="MS")
    filas = []
    for fecha in fechas:
        for depto in DEPARTAMENTOS:
            for linea in (4, LINEA_OBJETIVO_CODIGO):
                pred = 100 + rng.normal(0, 5)
                filas.append({"fecha": fecha, "departamento": depto, "linea_codigo": linea,
                              "y_pred": pred, "y_real": pred * (1 + rng.normal(0.01, 0.04))})
    return pd.DataFrame(filas)


def test_rango_error_formato_y_orden_de_percentiles():
    rango = rango_error(_oof_inventado())
    assert list(rango.columns) == COLUMNAS_RANGO
    # 7 departamentos × 2 líneas + Nacional de cada línea
    assert len(rango) == len(DEPARTAMENTOS) * 2 + 2
    assert set(rango["ambito"]) == {*DEPARTAMENTOS, NACIONAL}
    assert (rango["p10"] <= rango["p50"]).all() and (rango["p50"] <= rango["p90"]).all()


def test_rango_error_a_mano():
    """Un departamento, una línea, 10 meses con error relativo 0,00, 0,01, ..., 0,09."""
    errores = np.arange(10) / 100
    df = pd.DataFrame({
        "fecha": pd.date_range("2030-01-01", periods=10, freq="MS"),
        "departamento": "Bogota",
        "linea_codigo": LINEA_OBJETIVO_CODIGO,
        "y_pred": 100.0,
        "y_real": 100.0 * (1 + errores),
    })
    rango = rango_error(df).set_index("ambito")
    assert rango.loc["Bogota", "p50"] == pytest.approx(np.quantile(errores, 0.5))
    assert rango.loc["Bogota", "p10"] == pytest.approx(np.quantile(errores, 0.1))
    # Con un solo departamento, el error nacional es el mismo (se agrega y se divide)
    assert rango.loc[NACIONAL, "p90"] == pytest.approx(rango.loc["Bogota", "p90"])


def test_rango_error_rechaza_varios_modelos():
    df = _oof_inventado()
    df["modelo"] = np.where(df.index % 2 == 0, "ridge", "knn")
    with pytest.raises(ValueError, match="un solo modelo"):
        rango_error(df)

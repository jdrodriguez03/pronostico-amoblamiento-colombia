# Dueño: Julián
"""Pruebas del StepwiseSelector con datos sintéticos de resultado conocido.

Se usan datos inventados (no la base) porque así sabemos de antemano la respuesta
correcta: qué columnas tienen señal y cuáles son ruido puro.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.modelos.stepwise import StepwiseSelector, tabla_historial

SEMILLA_PRUEBA = 7


def datos_sinteticos(n=400, semilla=SEMILLA_PRUEBA):
    """y depende de señal_1 y señal_2; ruido_1..3 no aportan; dummy_a/b son 'forzadas'."""
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame({
        "cat__dummy_a": rng.integers(0, 2, n),   # forzada sin efecto real
        "cat__dummy_b": rng.integers(0, 2, n),   # forzada sin efecto real
        "num__senal_1": rng.normal(size=n),
        "num__senal_2": rng.normal(size=n),
        "num__ruido_1": rng.normal(size=n),
        "num__ruido_2": rng.normal(size=n),
        "num__ruido_3": rng.normal(size=n),
    }).astype(float)
    y = 2.0 * X["num__senal_1"] - 1.5 * X["num__senal_2"] + rng.normal(scale=1.0, size=n)
    return X, y


def elegidas(selector):
    return list(selector.get_feature_names_out())


def test_la_senal_entra_y_el_ruido_no():
    X, y = datos_sinteticos()
    # alfa estricto (0,01): con un alfa de 0,25 cada columna de ruido puro tiene ~25 %
    # de probabilidad de entrar por azar, que es justo la crítica al stepwise.
    sel = StepwiseSelector(alfa_entrada=0.01, alfa_salida=0.01, forzadas=["cat__"]).fit(X, y)
    assert "num__senal_1" in elegidas(sel)
    assert "num__senal_2" in elegidas(sel)
    assert not any(c.startswith("num__ruido") for c in elegidas(sel))


def test_las_forzadas_siempre_quedan_aunque_no_aporten():
    X, y = datos_sinteticos()
    sel = StepwiseSelector(forzadas=["cat__"]).fit(X, y)
    assert {"cat__dummy_a", "cat__dummy_b"} <= set(elegidas(sel))


def test_forzadas_por_posicion_con_numpy():
    X, y = datos_sinteticos()
    sel = StepwiseSelector(forzadas=[0, 1]).fit(X.to_numpy(), y.to_numpy())
    assert sel.get_support()[:2].all()
    assert sel.get_support()[2:4].all()  # las dos señales


def test_forzadas_por_nombre_sin_dataframe_falla():
    X, y = datos_sinteticos()
    with pytest.raises(ValueError, match="DataFrame"):
        StepwiseSelector(forzadas=["cat__"]).fit(X.to_numpy(), y)


def test_alfas_incoherentes_fallan():
    X, y = datos_sinteticos()
    with pytest.raises(ValueError, match="alfa_entrada"):
        StepwiseSelector(alfa_entrada=0.3, alfa_salida=0.1).fit(X, y)


def test_historial_registra_las_entradas():
    X, y = datos_sinteticos()
    sel = StepwiseSelector(alfa_entrada=0.01, alfa_salida=0.01, forzadas=["cat__"]).fit(X, y)
    hist = tabla_historial(sel)
    assert list(hist.columns) == ["accion", "variable", "p_valor"]
    # La primera en entrar es la de señal más fuerte
    assert hist.iloc[0].tolist()[:2] == ["entra", "num__senal_1"]


def test_funciona_dentro_de_un_pipeline_y_se_puede_clonar():
    X, y = datos_sinteticos()
    pipe = Pipeline([
        ("sel", StepwiseSelector(alfa_entrada=0.01, alfa_salida=0.01, forzadas=["cat__"])),
        ("modelo", LinearRegression()),
    ])
    pipe = clone(pipe).fit(X, y)  # clone es lo que hace GridSearchCV en cada fold
    assert pipe.named_steps["modelo"].coef_.shape == (4,)  # 2 forzadas + 2 señales
    assert pipe.predict(X).shape == (len(X),)

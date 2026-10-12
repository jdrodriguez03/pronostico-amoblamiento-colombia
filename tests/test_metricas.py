# Dueño: Juanes
"""Pruebas de las métricas con un ejemplo de 3 valores calculado a mano.

y    = [100, 200, 400]
yhat = [110, 190, 400]
e    = [-10,  10,   0]

MAPE = 100 · (10/100 + 10/200 + 0/400) / 3 = 100 · 0,15 / 3 = 5 %
MAD  = (10 + 10 + 0) / 3 = 20/3 ≈ 6,6667
MSD  = (100 + 100 + 0) / 3 = 200/3 ≈ 66,6667
"""

import pytest

from src.evaluacion.metricas import mad, mape, mejora_pct, msd

Y = [100, 200, 400]
YHAT = [110, 190, 400]


def test_mape_a_mano():
    assert mape(Y, YHAT) == pytest.approx(5.0)


def test_mad_a_mano():
    assert mad(Y, YHAT) == pytest.approx(20 / 3)


def test_msd_a_mano():
    assert msd(Y, YHAT) == pytest.approx(200 / 3)


def test_prediccion_perfecta_da_cero():
    assert mape(Y, Y) == 0
    assert mad(Y, Y) == 0
    assert msd(Y, Y) == 0


def test_mejora_pct():
    assert mejora_pct(8, 10) == pytest.approx(20.0)   # el modelo se equivoca menos
    assert mejora_pct(12, 10) == pytest.approx(-20.0)  # el modelo lo hace peor
    assert mejora_pct(10, 10) == 0


def test_errores_claros():
    with pytest.raises(ValueError):
        mape([0, 100], [1, 100])  # MAPE indefinido con y = 0
    with pytest.raises(ValueError):
        mad([1, 2, 3], [1, 2])  # tamaños distintos
    with pytest.raises(ValueError):
        mejora_pct(5, 0)

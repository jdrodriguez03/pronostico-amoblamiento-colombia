# Dueño: Juanes
"""Pruebas de las métricas con un ejemplo de 3 valores calculado a mano.

y    = [100, 200, 400]
yhat = [110, 190, 400]
e    = [-10,  10,   0]

MAPE = 100 · (10/100 + 10/200 + 0/400) / 3 = 100 · 0,15 / 3 = 5 %
MAD  = (10 + 10 + 0) / 3 = 20/3 ≈ 6,6667
MSD  = (100 + 100 + 0) / 3 = 200/3 ≈ 66,6667
RMSE = √(200/3) ≈ 8,1650                 (mayor que el MAD: el 0 no "compensa" en cuadrados)
"""

import pytest

from src.evaluacion.metricas import mad, mape, mejora_pct, msd, rmse

Y = [100, 200, 400]
YHAT = [110, 190, 400]


def test_mape_a_mano():
    assert mape(Y, YHAT) == pytest.approx(5.0)


def test_mad_a_mano():
    assert mad(Y, YHAT) == pytest.approx(20 / 3)


def test_msd_a_mano():
    assert msd(Y, YHAT) == pytest.approx(200 / 3)


def test_rmse_a_mano():
    assert rmse(Y, YHAT) == pytest.approx(8.164966, abs=1e-6)  # √(200/3), con calculadora
    assert rmse(Y, YHAT) ** 2 == pytest.approx(msd(Y, YHAT))   # RMSE² = MSD
    assert rmse(Y, YHAT) >= mad(Y, YHAT)                       # siempre RMSE ≥ MAD


def test_rmse_igual_a_mad_si_todos_los_errores_miden_lo_mismo():
    # Errores −3, +3, −3, +3: todos de tamaño 3, así que RMSE = MAD = 3
    y, yhat = [100, 100, 100, 100], [103, 97, 103, 97]
    assert rmse(y, yhat) == pytest.approx(3.0)
    assert mad(y, yhat) == pytest.approx(3.0)


def test_prediccion_perfecta_da_cero():
    assert mape(Y, Y) == 0
    assert mad(Y, Y) == 0
    assert msd(Y, Y) == 0
    assert rmse(Y, Y) == 0


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

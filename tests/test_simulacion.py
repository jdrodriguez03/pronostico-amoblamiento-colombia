import numpy as np
import pytest

from src.config import DEPARTAMENTOS, ESCENARIOS, LINEAS, N_MESES, PESOS_DEPTO, RHO, SEMILLA
from src.datos.simulacion import conservar, simular_eta


@pytest.mark.parametrize("sigma", ESCENARIOS.values(), ids=ESCENARIOS.keys())
def test_simulacion_es_reproducible(sigma):
    primero = simular_eta(sigma, semilla=SEMILLA)
    segundo = simular_eta(sigma, semilla=SEMILLA)
    np.testing.assert_array_equal(primero, segundo)
    assert primero.shape == (N_MESES, len(DEPARTAMENTOS), len(LINEAS))


def test_conservacion_reproduce_agregado_nacional():
    rng = np.random.default_rng(SEMILLA)
    eta = np.exp(rng.normal(0, 0.07, (N_MESES, len(DEPARTAMENTOS), len(LINEAS))))
    F = rng.uniform(0.7, 1.3, (N_MESES, len(DEPARTAMENTOS)))
    pesos = np.array([PESOS_DEPTO[d] for d in DEPARTAMENTOS])
    F /= np.einsum("d,td->t", pesos, F)[:, None]

    ajustada = conservar(eta, F)
    agregado = np.einsum("d,td,tdl->tl", pesos, F, ajustada)
    np.testing.assert_allclose(agregado, 1.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("sigma", ESCENARIOS.values(), ids=ESCENARIOS.keys())
def test_variabilidad_logaritmica_estacionaria(sigma):
    eta = simular_eta(sigma)
    desviacion = np.log(eta).std()
    assert 0.8 * sigma <= desviacion <= 1.2 * sigma


@pytest.mark.parametrize("sigma", ESCENARIOS.values(), ids=ESCENARIOS.keys())
def test_autocorrelacion_ar1(sigma):
    eta = simular_eta(sigma)
    log_eta = np.log(eta)
    autocorrelaciones = [
        np.corrcoef(log_eta[:-1, d, linea_idx], log_eta[1:, d, linea_idx])[0, 1]
        for d in range(len(DEPARTAMENTOS))
        for linea_idx in range(len(LINEAS))
    ]
    assert abs(np.mean(autocorrelaciones) - RHO) <= 0.1

"""Simulación del componente específico por departamento y línea."""

import numpy as np

from src.config import DEPARTAMENTOS, LINEAS, N_MESES, PESOS_DEPTO, RHO, SEMILLA


def simular_eta(sigma: float, phi: float = RHO, semilla: int = SEMILLA) -> np.ndarray:
    """Genera eta con forma (mes, departamento, línea) y desviación logarítmica sigma.

    El primer estado se toma de la distribución estacionaria. Se centra la
    exponencial como en Entrega 1 para que su esperanza sea uno.
    """
    if sigma < 0:
        raise ValueError("sigma debe ser no negativo")
    if not -1 < phi < 1:
        raise ValueError("phi debe estar entre -1 y 1 para un AR(1) estacionario")

    rng = np.random.default_rng(semilla)
    shape = (N_MESES, len(DEPARTAMENTOS), len(LINEAS))
    e = np.empty(shape, dtype=float)
    e[0] = rng.normal(0.0, sigma, shape[1:])
    escala = sigma * np.sqrt(1.0 - phi**2)
    for t in range(1, N_MESES):
        e[t] = phi * e[t - 1] + rng.normal(0.0, escala, shape[1:])
    return np.exp(e - sigma**2 / 2)


def conservar(eta: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Ajusta eta para que sum_d(peso_d F[d] eta[d]) sea uno por mes y línea.

    `eta` sigue el orden (mes, departamento, línea); `F` tiene forma
    (mes, departamento), como el factor regional producido por la EMC.
    Devuelve un arreglo nuevo. Este ajuste conserva la serie nacional cuando
    el índice regional es L * F * eta y el agregado ponderado de F es uno.
    """
    eta = np.asarray(eta, dtype=float)
    F = np.asarray(F, dtype=float)
    pesos = np.array([PESOS_DEPTO[d] for d in DEPARTAMENTOS], dtype=float)
    if eta.ndim != 3 or eta.shape[1:] != (len(DEPARTAMENTOS), len(LINEAS)):
        raise ValueError("eta debe tener forma (meses, 7 departamentos, 8 líneas)")
    if F.shape != eta.shape[:2]:
        raise ValueError("F debe tener forma (meses, 7 departamentos)")
    agregado = np.einsum("d,td,tdl->tl", pesos, F, eta)
    if not np.isfinite(agregado).all() or (agregado == 0).any():
        raise ValueError("el agregado ponderado debe ser finito y distinto de cero")
    return eta / agregado[:, None, :]

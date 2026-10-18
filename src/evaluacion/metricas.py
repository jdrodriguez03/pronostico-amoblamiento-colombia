# Dueño: Juanes
"""MAPE, MAD y MSD (las métricas del artículo) y mejora frente a una línea base.

Día 3 de la guía. Son las tres medidas que reporta el artículo base (MAPE 4,89 %,
MAD 447,14 y MSD 417.422,39 para su MLR), así que se calculan con la misma definición
para poder comparar:

    e_t  = y_t − ŷ_t                          (error de cada mes)
    MAPE = 100 · promedio(|e_t| / |y_t|)       en %: no depende de la escala
    MAD  = promedio(|e_t|)                     en unidades del índice
    MSD  = promedio(e_t²)                      castiga más los errores grandes

Todas se calculan en la ESCALA ORIGINAL del índice (no en logaritmo): el Pipeline
modela log(y), pero TransformedTargetRegressor devuelve las predicciones ya en escala
original.
"""

import numpy as np


def _como_arreglos(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    """Convierte a arreglos de números y revisa que sean comparables."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    if y.shape != yhat.shape:
        raise ValueError(f"y tiene {y.size} valores y yhat {yhat.size}")
    if y.size == 0:
        raise ValueError("no hay valores para calcular la métrica")
    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("hay valores vacíos (NaN) en y o en yhat")
    return y, yhat


def mape(y, yhat) -> float:
    """Error porcentual absoluto medio, en %."""
    y, yhat = _como_arreglos(y, yhat)
    if (y == 0).any():
        # Con y = 0 el MAPE se indefine; por eso se reporta MAD como contraste
        raise ValueError("el MAPE no está definido cuando algún valor observado es 0")
    return float(np.mean(np.abs(y - yhat) / np.abs(y)) * 100)


def mad(y, yhat) -> float:
    """Desviación absoluta media (error absoluto medio)."""
    y, yhat = _como_arreglos(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def msd(y, yhat) -> float:
    """Desviación cuadrática media (error cuadrático medio)."""
    y, yhat = _como_arreglos(y, yhat)
    return float(np.mean((y - yhat) ** 2))


def mejora_pct(metrica_modelo, metrica_base) -> float:
    """Cuánto mejora (en %) el modelo frente a la base.

    Las tres métricas son de error (menos es mejor), así que:
        mejora = 100 · (base − modelo) / base
    Positivo: el modelo se equivoca menos que la base. Negativo: lo hace peor.
    Ejemplo: base 10 %, modelo 8 % → mejora del 20 %.
    """
    if metrica_base == 0:
        raise ValueError("la métrica de la base es 0: no se puede calcular la mejora relativa")
    return float(100 * (metrica_base - metrica_modelo) / metrica_base)

# Dueño: Julián
import numpy as np
import pandas as pd
import pytest

from src.config import SEMILLA, predictores_modelo
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.interpretacion import calcular_vif


def _datos_sinteticos(n: int = 500) -> pd.DataFrame:
    """Tres columnas independientes y una cuarta construida a partir de la primera."""
    rng = np.random.default_rng(SEMILLA)
    x1, x2, x3 = rng.normal(size=(3, n))
    x4 = 0.8 * x1 + 0.6 * rng.normal(size=n)  # correlacionada con x1, pero no idéntica
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})


def _vif_a_mano(df: pd.DataFrame, columna: str) -> float:
    """1 / (1 - R²) de regresar `columna` contra las demás, con intercepto (mínimos cuadrados)."""
    y = df[columna].to_numpy()
    otras = df.drop(columns=columna).to_numpy()
    A = np.column_stack([np.ones(len(df)), otras])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    residuo = y - A @ beta
    r2 = 1 - residuo.var() / y.var()
    return 1 / (1 - r2)


def test_vif_columnas_independientes_da_cerca_de_uno():
    df = _datos_sinteticos()[["x1", "x2", "x3"]]
    vif = calcular_vif(df, ["x1", "x2", "x3"])
    assert (vif["vif"] < 1.1).all()


def test_vif_coincide_con_la_formula_a_mano():
    df = _datos_sinteticos()
    vif = calcular_vif(df, list(df.columns)).set_index("variable")["vif"]
    for columna in df.columns:
        assert vif[columna] == pytest.approx(_vif_a_mano(df, columna), rel=1e-8)


def test_vif_detecta_colinealidad_fuerte():
    df = _datos_sinteticos()
    # Ruido con otra semilla: con la misma saldría idéntico a x1 y la colinealidad sería perfecta
    ruido = np.random.default_rng(SEMILLA + 1).normal(size=len(df))
    df["x5"] = df["x1"] + 0.05 * ruido  # casi igual a x1, pero no exacta
    vif = calcular_vif(df, ["x1", "x2", "x5"]).set_index("variable")["vif"]
    assert vif["x1"] > 100 and vif["x5"] > 100
    assert vif["x2"] < 1.1


def test_vif_respeta_orden_y_nombres():
    df = _datos_sinteticos()
    vif = calcular_vif(df, ["x3", "x1"])
    assert vif["variable"].tolist() == ["x3", "x1"]
    assert list(vif.columns) == ["variable", "vif"]


def test_vif_no_admite_faltantes():
    df = _datos_sinteticos()
    df.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError, match="faltantes"):
        calcular_vif(df, ["x1", "x2"])


def test_vif_en_la_base_real_es_valido():
    """Humo: corre con los predictores reales y cada VIF es finito y al menos 1."""
    df = cargar_entrenamiento("central")
    vif = calcular_vif(df, predictores_modelo())
    assert len(vif) == 5
    assert np.isfinite(vif["vif"]).all()
    assert (vif["vif"] >= 1).all()

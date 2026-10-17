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


# ---------------------------------------------------------------------------
# coeficientes() y estabilidad()
# ---------------------------------------------------------------------------
def _pipeline_lineal(modelo):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    prep = ColumnTransformer([("num", StandardScaler(), ["x1", "x2"])])
    return Pipeline([("prep", prep), ("modelo", modelo)])


def test_coeficientes_tienen_nombres_y_valores_estandarizados():
    from sklearn.linear_model import LinearRegression

    from src.evaluacion.interpretacion import coeficientes

    rng = np.random.default_rng(SEMILLA)
    df = pd.DataFrame({"x1": rng.normal(10, 5, 300), "x2": rng.normal(0, 1, 300)})
    z = (df - df.mean()) / df.std(ddof=0)  # lo mismo que hace StandardScaler
    y = 3 * z["x1"] - 2 * z["x2"]  # 3 y -2 por desviación estándar
    tabla = coeficientes(_pipeline_lineal(LinearRegression()).fit(df, y))
    assert list(tabla["variable"]) == ["num__x1", "num__x2"]
    assert np.allclose(tabla["coeficiente"], [3, -2])
    assert set(tabla["tipo"]) == {"numerica"}


def test_coeficientes_rechaza_knn():
    from sklearn.neighbors import KNeighborsRegressor

    from src.evaluacion.interpretacion import coeficientes

    df = pd.DataFrame({"x1": range(10), "x2": range(10)}, dtype=float)
    with pytest.raises(TypeError, match="KNN"):
        coeficientes(_pipeline_lineal(KNeighborsRegressor(n_neighbors=2)).fit(df, df["x1"]))


def test_estabilidad_media_desviacion_y_signo():
    from src.evaluacion.interpretacion import estabilidad

    por_fold = pd.DataFrame({
        "fold": [1, 2, 3, 1, 2, 3, 1, 3],
        "variable": ["a", "a", "a", "b", "b", "b", "c", "c"],
        "coeficiente": [1.0, 2.0, 3.0, 0.5, -0.5, 0.0, 4.0, 4.0],
    })
    t = estabilidad(por_fold).set_index("variable")
    assert t.loc["a", "media"] == pytest.approx(2.0)
    assert t.loc["a", "desviacion"] == pytest.approx(1.0)
    assert t.loc["a", "signo_estable"]
    assert not t.loc["b", "signo_estable"]          # cambia de signo
    assert t.loc["b", "folds_distinto_de_cero"] == 2
    assert t.loc["c", "folds_distinto_de_cero"] == 2  # falta en el fold 2: cuenta como 0
    assert t.loc["c", "media"] == pytest.approx(8 / 3)


class _CortesPorMes:
    """Tres folds de ventana creciente que nunca parten un mes (regla 1 del repo)."""

    def split(self, df):
        meses = np.sort(df["fecha"].unique())
        for corte in meses[[48, 60, 72]]:
            yield np.where(df["fecha"] < corte)[0], np.where(df["fecha"] == corte)[0]


def test_coeficientes_por_fold_con_la_base_real():
    from sklearn.linear_model import Ridge

    from src.config import OBJETIVO
    from src.evaluacion.interpretacion import coeficientes_por_fold, estabilidad

    df = cargar_entrenamiento("central")
    df = df.assign(x1=df[predictores_modelo()[0]], x2=df[predictores_modelo()[1]])
    por_fold = coeficientes_por_fold(
        _pipeline_lineal(Ridge(alpha=1.0)), df, _CortesPorMes(), OBJETIVO
    )
    assert sorted(por_fold["fold"].unique()) == [1, 2, 3]
    assert len(estabilidad(por_fold)) == 2

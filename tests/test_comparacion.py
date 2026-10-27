# Dueño: Juanes
"""Pruebas de las tablas de comparación con un cv_resultados de juguete (cuentas a mano)."""

import pandas as pd
import pytest

from src.evaluacion.comparacion import tabla_comparacion, tabla_pareada


def _cv_juguete() -> pd.DataFrame:
    """2 escenarios × 3 modelos × 3 folds con MAPE elegidos para hacer las cuentas a mano.

    Escenario "central":
        media               10, 10, 10   → media 10
        ingenuo_estacional   8, 12, 10   → media 10
        ridge                6, 13,  5   → media 8  (gana al ingenuo en los folds 1 y 3)
    """
    mape = {
        "media": [10, 10, 10],
        "ingenuo_estacional": [8, 12, 10],
        "ridge": [6, 13, 5],
    }
    filas = []
    for escenario, factor in [("central", 1.0), ("alto", 2.0)]:
        for modelo, valores in mape.items():
            for fold, v in enumerate(valores, start=1):
                filas.append({
                    "escenario": escenario, "modelo": modelo, "fold": fold,
                    "hiperparametros": "alpha=1" if modelo == "ridge" else "sin hiperparámetros",
                    "mape": v * factor, "mad": v, "msd": v**2, "rmse": v,
                    "mape_entrena": v / 2, "mape_ajuste": 3.0,
                })
    return pd.DataFrame(filas)


def test_comparacion_una_fila_por_escenario_y_modelo_ordenada():
    tabla = tabla_comparacion(_cv_juguete())
    assert len(tabla) == 6
    central = tabla[tabla["escenario"] == "central"]
    # Ordenada de mejor a peor MAPE medio; ridge primero
    assert central["modelo"].iloc[0] == "ridge"
    assert central["mape_media"].iloc[0] == pytest.approx(8.0)


def test_comparacion_desviacion_y_mejoras_a_mano():
    tabla = tabla_comparacion(_cv_juguete()).set_index(["escenario", "modelo"])
    ridge = tabla.loc[("central", "ridge")]
    # Desviación muestral de (6, 13, 5): media 8, suma de cuadrados 4 + 25 + 9 = 38, /2 = 19
    assert ridge["mape_desv"] == pytest.approx(19**0.5)
    # Mejora frente a la media y al ingenuo (los dos con MAPE medio 10): 100·(10 − 8)/10
    assert ridge["mejora_vs_media"] == pytest.approx(20.0)
    assert ridge["mejora_vs_ingenuo"] == pytest.approx(20.0)
    assert ridge["mape_entrena_media"] == pytest.approx(4.0)
    assert ridge["mape_ajuste"] == pytest.approx(3.0)
    assert ridge["hiperparametros"] == "alpha=1"
    # Cada línea base mejora 0 % frente a sí misma
    assert tabla.loc[("central", "media"), "mejora_vs_media"] == pytest.approx(0.0)


def test_comparacion_compara_dentro_del_mismo_escenario():
    """En "alto" todos los MAPE se duplican: las mejoras relativas no cambian."""
    tabla = tabla_comparacion(_cv_juguete()).set_index(["escenario", "modelo"])
    assert tabla.loc[("alto", "ridge"), "mape_media"] == pytest.approx(16.0)
    assert tabla.loc[("alto", "ridge"), "mejora_vs_ingenuo"] == pytest.approx(20.0)


def test_pareada_diferencias_fold_a_fold_a_mano():
    pareada = tabla_pareada(_cv_juguete(), referencias=("ingenuo_estacional",))
    fila = pareada[(pareada["escenario"] == "central") & (pareada["modelo"] == "ridge")]
    fila = fila.iloc[0]
    # ridge − ingenuo: 6 − 8 = −2, 13 − 12 = 1, 5 − 10 = −5
    assert [fila["dif_f1"], fila["dif_f2"], fila["dif_f3"]] == [-2, 1, -5]
    assert fila["dif_media"] == pytest.approx(-2.0)
    assert fila["gana_en"] == 2
    assert fila["pierde_en"] == 1
    # El modelo no se compara contra sí mismo
    assert "ingenuo_estacional" not in set(pareada["modelo"])


def test_pareada_omite_referencias_que_no_estan():
    """mlr_stepwise no está en el juguete: no hay filas contra él, y no falla."""
    pareada = tabla_pareada(_cv_juguete())
    assert set(pareada["referencia"]) == {"ingenuo_estacional", "media"}


def test_pareada_contra_la_media_cuentas_a_mano():
    """Referencia media = 10, 10, 10 (escenario central del juguete).

    ridge    6, 13, 5  → diferencias −4, +3, −5: gana en 2, pierde en 1, media −2
    ingenuo  8, 12, 10 → diferencias −2, +2,  0: gana en 1, pierde en 1 (el empate no cuenta)
    """
    pareada = tabla_pareada(_cv_juguete())
    central = pareada[(pareada["escenario"] == "central") & (pareada["referencia"] == "media")]
    ridge = central.set_index("modelo").loc["ridge"]
    assert (ridge["gana_en"], ridge["pierde_en"]) == (2, 1)
    assert ridge["dif_media"] == pytest.approx(-2)
    ingenuo = central.set_index("modelo").loc["ingenuo_estacional"]
    assert (ingenuo["gana_en"], ingenuo["pierde_en"]) == (1, 1)


def test_rechaza_folds_distintos_entre_modelos():
    cv = _cv_juguete()
    cv = cv[~((cv["modelo"] == "ridge") & (cv["fold"] == 3))]
    with pytest.raises(ValueError, match="mismos folds"):
        tabla_pareada(cv)


def test_rechaza_si_falta_una_linea_base():
    cv = _cv_juguete()
    with pytest.raises(ValueError, match="media"):
        tabla_comparacion(cv[cv["modelo"] != "media"])


# ---------------------------------------------------------------------------
# tabla_rango_escenarios (martes 27)
# ---------------------------------------------------------------------------
def test_rango_escenarios_cuentas_a_mano():
    """Con el juguete, en "alto" todos los MAPE valen el doble que en "central".

    central: ridge 8, media 10, ingenuo 10   →  alto: ridge 16, media 20, ingenuo 20
    Rango de ridge = 16 − 8 = 8; de media e ingenuo = 20 − 10 = 10.
    """
    from src.evaluacion.comparacion import tabla_rango_escenarios

    rango = tabla_rango_escenarios(tabla_comparacion(_cv_juguete())).set_index("modelo")
    assert list(rango.index)[0] == "ridge"  # ordenada por el escenario central
    assert rango.loc["ridge", "mape_central"] == pytest.approx(8)
    assert rango.loc["ridge", "mape_alto"] == pytest.approx(16)
    assert rango.loc["ridge", "rango"] == pytest.approx(8)
    assert rango.loc["media", "rango"] == pytest.approx(10)
    # ridge es el mejor en los dos escenarios; media e ingenuo empatan en el puesto 2
    assert rango.loc["ridge", "puesto_central"] == 1
    assert rango.loc["ridge", "puesto_alto"] == 1
    assert rango.loc["media", "puesto_central"] == 2
    assert rango.loc["ingenuo_estacional", "puesto_central"] == 2
    # Columnas de escenario en el orden de config, solo las que existen
    assert "mape_bajo" not in rango.columns


def test_rango_escenarios_detecta_cambio_de_puesto():
    """Si un modelo pasa del 1.º al 2.º puesto entre escenarios, la tabla lo muestra."""
    from src.evaluacion.comparacion import tabla_rango_escenarios

    comparacion = pd.DataFrame({
        "escenario": ["central", "central", "alto", "alto"],
        "modelo": ["knn", "ridge", "knn", "ridge"],
        "mape_media": [9.0, 9.1, 10.2, 9.1],
    })
    rango = tabla_rango_escenarios(comparacion).set_index("modelo")
    assert rango.loc["knn", "puesto_central"] == 1
    assert rango.loc["knn", "puesto_alto"] == 2
    assert rango.loc["knn", "rango"] == pytest.approx(1.2)
    assert rango.loc["ridge", "rango"] == pytest.approx(0.0)


def test_rango_escenarios_falla_sin_columnas():
    from src.evaluacion.comparacion import tabla_rango_escenarios

    with pytest.raises(ValueError, match="mape_media"):
        tabla_rango_escenarios(pd.DataFrame({"escenario": ["central"], "modelo": ["knn"]}))

# Dueño: Juanes
"""Prueba de scripts/evaluar_prueba.py SIN abrir la prueba (entrega final, lunes 9).

La regla 1 de la entrega prohíbe calcular cualquier cifra con agosto 2025 – julio 2026 antes
del martes. Por eso aquí la "prueba" es INVENTADA: se toman los últimos 12 meses del
entrenamiento de la base pequeña, se corren 12 meses hacia adelante y se les cambia el
objetivo. Así se ejercita todo el camino del script (rezagos sobre la serie completa,
reentrenar, predecir, tablas, regla) sin leer ni un dato real de la prueba: cargar_prueba se
reemplaza con monkeypatch y nunca se llama.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import INICIO_PRUEBA, LINEA_OBJETIVO_CODIGO, OBJETIVO
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.comparacion import regla_modelo_final
from src.features.rezagos import columnas_predictoras
from src.modelos.pipeline import preparar_datos

RUTA_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluar_prueba.py"


def _cargar_script():
    spec = importlib.util.spec_from_file_location("evaluar_prueba", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _prueba_inventada(entrena: pd.DataFrame) -> pd.DataFrame:
    """Últimos 12 meses del entrenamiento corridos 12 meses: ago 2025 – jul 2026, inventados."""
    ultimos = entrena[entrena["fecha"] >= entrena["fecha"].max() - pd.DateOffset(months=11)]
    falsa = ultimos.copy()
    falsa["fecha"] = falsa["fecha"] + pd.DateOffset(months=12)
    falsa["anio"] = falsa["fecha"].dt.year
    falsa[OBJETIVO] = falsa[OBJETIVO] * 1.03
    return falsa.reset_index(drop=True)


@pytest.fixture(scope="module")
def datos():
    entrena = cargar_entrenamiento(fixture=True)
    return entrena, _prueba_inventada(entrena)


@pytest.fixture(scope="module")
def corrida(datos, tmp_path_factory):
    script = _cargar_script()
    entrena, prueba = datos
    carpeta = tmp_path_factory.mktemp("prueba")
    mp = pytest.MonkeyPatch()
    mp.setattr(script, "RUTA_SALIDA", carpeta)
    mp.setattr(script, "cargar_entrenamiento", lambda escenario: entrena.copy())
    mp.setattr(script, "cargar_prueba", lambda escenario: prueba.copy())
    try:
        tablas = script.main(["--escenario", "central"])
    finally:
        mp.undo()
    return script, carpeta, tablas, mp


def test_la_prueba_inventada_no_es_la_real(datos):
    """Guarda de la guarda: las fechas sí son las de la prueba, los valores no son reales."""
    _, prueba = datos
    assert prueba["fecha"].min() == pd.Timestamp(INICIO_PRUEBA)
    assert prueba["fecha"].nunique() == 12


def test_separar_con_rezagos_toma_el_rezago_del_entrenamiento(datos):
    script = _cargar_script()
    entrena, prueba = datos
    ent, pru, predictores = script.separar_con_rezagos(entrena, prueba)
    assert predictores == columnas_predictoras()
    # La prueba conserva sus 12 meses: el rezago de agosto sale de mayo-julio de entrenamiento
    assert len(pru) == len(prueba) and pru[predictores].notna().all().all()
    agosto = pru[pru["fecha"] == pd.Timestamp(INICIO_PRUEBA)].iloc[0]
    fila_mayo = entrena[(entrena["fecha"] == pd.Timestamp("2025-05-01"))
                        & (entrena["departamento"] == agosto["departamento"])
                        & (entrena["linea_codigo"] == agosto["linea_codigo"])].iloc[0]
    assert agosto["importaciones_muebles_usd_miles_rez3"] == fila_mayo[
        "importaciones_muebles_usd_miles"]
    # El entrenamiento queda IGUAL al de entrenar.py: mismos hiperparámetros garantizados
    solo_ent, _ = preparar_datos(entrena, "reales")
    pd.testing.assert_frame_equal(ent[solo_ent.columns], solo_ent)
    assert pd.to_datetime(ent["fecha"]).max() < pd.Timestamp(INICIO_PRUEBA)


def test_separar_rechaza_particiones_cruzadas(datos):
    script = _cargar_script()
    entrena, prueba = datos
    with pytest.raises(ValueError, match="entrenamiento trae meses"):
        script.separar_con_rezagos(pd.concat([entrena, prueba]), prueba)


def test_tablas_con_el_formato_del_contrato(corrida):
    script, carpeta, tablas, _ = corrida
    for nombre, columnas in [("prueba", script.COLUMNAS_PRUEBA),
                             ("prueba_predicciones", script.COLUMNAS_PREDICCIONES),
                             ("prueba_pareada", script.COLUMNAS_PAREADA)]:
        leido = pd.read_csv(carpeta / f"{nombre}.csv")
        assert list(leido.columns) == columnas
    prueba = tablas["prueba"]
    assert set(prueba["modelo"]) == {"ridge", "knn", "mlr_stepwise", "media",
                                     "ingenuo_estacional"}
    assert (prueba["n_meses"] == 12).all()
    assert prueba[["mape", "mad", "rmse", "mape_panel", "mape_cv_desv"]].notna().all().all()
    # Una línea base contra sí misma mejora 0 %
    fila = prueba.set_index("modelo")
    assert fila.loc["media", "mejora_vs_media"] == pytest.approx(0)
    assert fila.loc["ingenuo_estacional", "mejora_vs_ingenuo"] == pytest.approx(0)


def test_predicciones_incluyen_nacional_de_la_linea_8(corrida):
    _, _, tablas, _ = corrida
    pred = tablas["prueba_predicciones"]
    nacional = pred[pred["ambito"] == "Nacional"]
    assert set(nacional["linea_codigo"]) == {LINEA_OBJETIVO_CODIGO}
    assert nacional.groupby("modelo")["fecha"].nunique().eq(12).all()
    # El MAPE de prueba.csv sale exactamente de esas filas nacionales
    ridge = nacional[nacional["modelo"] == "ridge"]
    mape_ridge = np.mean(np.abs(ridge["y_real"] - ridge["y_pred"]) / ridge["y_real"]) * 100
    fila = tablas["prueba"].set_index("modelo").loc["ridge"]
    assert fila["mape"] == pytest.approx(mape_ridge)


def test_regla_desde_tablas_coincide_con_la_pareada(corrida):
    script, _, tablas, _ = corrida
    par = tablas["prueba_pareada"]
    assert len(par) == 12
    r = script.regla_desde_tablas(tablas["prueba"], par, "central")
    assert r["meses_knn_gana"] == int(par["knn_gana"].sum())
    assert r["modelo_final"] in {"ridge", "knn"}


def test_no_corre_dos_veces_sin_forzar(corrida):
    script, carpeta, _, _ = corrida
    mp = pytest.MonkeyPatch()
    mp.setattr(script, "RUTA_SALIDA", carpeta)
    try:
        with pytest.raises(SystemExit, match="una sola vez"):
            script.main(["--escenario", "central"])
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# La regla congelada, con números a mano
# ---------------------------------------------------------------------------

def _serie(errores_ridge, errores_knn):
    y = np.full(12, 100.0)
    return y, y - np.asarray(errores_ridge), y - np.asarray(errores_knn)


def test_regla_se_queda_con_ridge_si_falta_una_condicion():
    # MAPE Ridge 8 % y KNN 4,75 %: gana por 3,25 puntos (> 0,9), (a) sí.
    # Pero KNN tiene menor error solo en 7 meses (en los otros 5 se equivoca 11,4): (b) no.
    y, r, k = _serie([8] * 12, [0] * 7 + [11.4] * 5)
    res = regla_modelo_final(y, r, k, desv_cv_ridge=0.9)
    assert res["condicion_a"] and not res["condicion_b"]
    assert res["meses_knn_gana"] == 7 and res["modelo_final"] == "ridge"


def test_regla_cambia_a_knn_solo_con_las_dos():
    y, r, k = _serie([8] * 12, [6] * 12)
    res = regla_modelo_final(y, r, k, desv_cv_ridge=0.9)
    assert res["condicion_a"] and res["condicion_b"] and res["modelo_final"] == "knn"
    # La misma ventaja con una desviación mayor ya no alcanza (desigualdad estricta)
    assert regla_modelo_final(y, r, k, desv_cv_ridge=2.0)["modelo_final"] == "ridge"


def test_regla_empate_de_un_mes_no_cuenta_para_knn():
    y, r, k = _serie([5] * 12, [5] * 12)
    assert regla_modelo_final(y, r, k, desv_cv_ridge=0.0)["meses_knn_gana"] == 0

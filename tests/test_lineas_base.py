# Dueño: Juanes
"""Pruebas de las líneas base sobre la base pequeña (fixture)."""

import pandas as pd
import pytest

from src.config import OBJETIVO
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.particion import CVTemporalPorMes
from src.modelos.lineas_base import IngenuoEstacional, PredictorMedia


@pytest.fixture(scope="module")
def mini():
    return cargar_entrenamiento(fixture=True)


def test_ingenuo_devuelve_exactamente_t_menos_12(mini):
    entrena, valida = next(iter(CVTemporalPorMes().split(mini)))
    ent, val = mini.iloc[entrena], mini.iloc[valida]
    pred = IngenuoEstacional().fit(ent, ent[OBJETIVO]).predict(val)

    # Lo mismo, buscado a mano fila por fila
    indice = mini.set_index(["departamento", "linea_codigo", "fecha"])[OBJETIVO]
    esperado = [
        indice[(f.departamento, f.linea_codigo, f.fecha - pd.DateOffset(months=12))]
        for f in val.itertuples()
    ]
    assert list(pred) == esperado  # exactamente, sin tolerancia


def test_ingenuo_falla_si_no_hay_t_menos_12(mini):
    primer_anio = mini[mini["fecha"] < "2020-02-01"]
    modelo = IngenuoEstacional().fit(primer_anio, primer_anio[OBJETIVO])
    with pytest.raises(ValueError, match="t−12"):
        modelo.predict(mini[mini["fecha"] == "2021-06-01"])


def test_media_es_el_promedio_de_cada_serie(mini):
    entrena, valida = next(iter(CVTemporalPorMes().split(mini)))
    ent, val = mini.iloc[entrena], mini.iloc[valida]
    pred = PredictorMedia().fit(ent, ent[OBJETIVO]).predict(val)

    medias = ent.groupby(["departamento", "linea_codigo"])[OBJETIVO].mean()
    for p, f in zip(pred, val.itertuples()):
        assert p == pytest.approx(medias[(f.departamento, f.linea_codigo)])
    # Cada serie tiene un único valor pronosticado (no cambia con el mes)
    assert val.assign(p=pred).groupby(["departamento", "linea_codigo"])["p"].nunique().eq(1).all()


def test_media_usa_solo_entrenamiento(mini):
    entrena, valida = next(iter(CVTemporalPorMes().split(mini)))
    ent, val = mini.iloc[entrena], mini.iloc[valida]
    pred = PredictorMedia().fit(ent, ent[OBJETIVO]).predict(val)
    media_con_todo = mini.groupby(["departamento", "linea_codigo"])[OBJETIVO].mean()
    primera = val.iloc[0]
    assert pred[0] != pytest.approx(media_con_todo[(primera.departamento, primera.linea_codigo)])

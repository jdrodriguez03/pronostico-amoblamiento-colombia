# Dueño: Juanes
"""Pruebas de CVTemporalPorMes (regla de oro 1: se parte por mes, nunca por fila)."""

import pandas as pd
import pytest

from src.config import INICIO_PRUEBA, RUTA_FIXTURE
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.particion import CVTemporalPorMes

# La tabla de docs/validacion.md: (fold, entrena hasta, meses de entrenamiento,
# valida desde, valida hasta). Todos los folds entrenan desde febrero de 2019.
TABLA_DOCS = [
    (1, "2023-01-01", 48, "2023-02-01", "2023-07-01"),
    (2, "2023-07-01", 54, "2023-08-01", "2024-01-01"),
    (3, "2024-01-01", 60, "2024-02-01", "2024-07-01"),
    (4, "2024-07-01", 66, "2024-08-01", "2025-01-01"),
    (5, "2025-01-01", 72, "2025-02-01", "2025-07-01"),
]


@pytest.fixture(scope="module")
def df():
    return cargar_entrenamiento("central")


def test_ningun_mes_en_entrenamiento_y_validacion(df):
    for entrena, valida in CVTemporalPorMes().split(df):
        meses_ent = set(df["fecha"].iloc[entrena])
        meses_val = set(df["fecha"].iloc[valida])
        assert not meses_ent & meses_val


def test_cada_mes_completo_de_un_solo_lado(df):
    # Las 56 filas de un mes de validación están todas en validación
    for _, valida in CVTemporalPorMes().split(df):
        assert len(valida) == 6 * 56


def test_validacion_siempre_posterior(df):
    for entrena, valida in CVTemporalPorMes().split(df):
        assert df["fecha"].iloc[entrena].max() < df["fecha"].iloc[valida].min()


def test_ventanas_coinciden_con_docs(df):
    ventanas = CVTemporalPorMes().ventanas(df)
    assert len(ventanas) == 5
    for fila, (fold, ent_hasta, n_ent, val_desde, val_hasta) in zip(
        ventanas.itertuples(), TABLA_DOCS
    ):
        assert fila.fold == fold
        assert fila.entrena_desde == pd.Timestamp("2019-02-01")
        assert fila.entrena_hasta == pd.Timestamp(ent_hasta)
        assert fila.meses_entrena == n_ent
        assert fila.valida_desde == pd.Timestamp(val_desde)
        assert fila.valida_hasta == pd.Timestamp(val_hasta)
        assert fila.meses_valida == 6


def test_ningun_mes_de_prueba(df):
    inicio = pd.Timestamp(INICIO_PRUEBA)
    for entrena, valida in CVTemporalPorMes().split(df):
        assert (df["fecha"].iloc[entrena] < inicio).all()
        assert (df["fecha"].iloc[valida] < inicio).all()


def test_rechaza_meses_de_prueba():
    # Si alguien le pasa la base completa, falla en vez de validar con la prueba
    completa = pd.read_csv(RUTA_FIXTURE, parse_dates=["fecha"])
    with pytest.raises(ValueError, match="prueba"):
        list(CVTemporalPorMes().split(completa))


def test_ingenuo_tiene_su_t_menos_12(df):
    # El mes t−12 de cada mes de validación está en el entrenamiento de ese fold
    for entrena, valida in CVTemporalPorMes().split(df):
        meses_ent = set(df["fecha"].iloc[entrena])
        for mes in set(df["fecha"].iloc[valida]):
            assert mes - pd.DateOffset(months=12) in meses_ent


def test_funciona_con_el_fixture():
    mini = cargar_entrenamiento(fixture=True)
    folds = list(CVTemporalPorMes().split(mini))
    assert len(folds) == 5
    assert all(len(v) == 6 * 4 for _, v in folds)  # 2 departamentos × 2 líneas

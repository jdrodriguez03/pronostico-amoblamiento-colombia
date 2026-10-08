# Dueño: Aleja
import pandas as pd

from src.config import INICIO_PRUEBA, USAR_REZAGO
from src.datos.cargar import cargar_entrenamiento, cargar_prueba


def test_entrenamiento_no_toca_la_prueba():
    df = cargar_entrenamiento("central")
    assert df["fecha"].max() < pd.Timestamp(INICIO_PRUEBA)
    assert df["fecha"].nunique() == (78 if USAR_REZAGO else 79)
    assert df["area_vivienda_m2_rez1"].notna().all() or not USAR_REZAGO


def test_prueba_son_12_meses():
    df = cargar_prueba("central")
    assert df["fecha"].min() == pd.Timestamp(INICIO_PRUEBA)
    assert df["fecha"].nunique() == 12
    assert len(df) == 12 * 56

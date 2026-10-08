# Dueño: Aleja
"""Carga del dataset ya partido en entrenamiento y prueba.

Regla de oro 3: cargar_prueba() SOLO se llama desde scripts/evaluar_prueba.py.
tests/test_fuga.py lo revisa. Día 4 de la guía.
"""

import pandas as pd


def cargar_entrenamiento(escenario: str = "central", fixture: bool = False) -> pd.DataFrame:
    """Meses de enero de 2019 a julio de 2025. Con fixture=True usa dataset_mini.parquet."""
    raise NotImplementedError("Aleja, día 4")


def cargar_prueba(escenario: str = "central") -> pd.DataFrame:
    """Meses de agosto de 2025 a julio de 2026. Solo para scripts/evaluar_prueba.py."""
    raise NotImplementedError("Aleja, día 4")

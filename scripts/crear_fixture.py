# Dueño: Juanes
"""Crea data/fixtures/base_mini.csv: un recorte pequeño de base_final_central.csv
(2 departamentos × 2 líneas, incluida la 8, × 91 meses = 364 filas) para pruebas rápidas.

Día 2 de la guía.

Por qué existe: las pruebas automáticas (pytest y el CI) deben correr en segundos y
probar la lógica, no la base completa de 5.096 filas. Con Antioquia y Bogota, las líneas
4 (prendas) y 8 (muebles) y los 91 meses, el recorte conserva todo lo que el código debe
manejar: varios departamentos, la línea objetivo y otra que no lo es, el mes sin rezago
(enero de 2019) y los meses de prueba (que cargar_entrenamiento debe filtrar).

Uso (desde la raíz del repo):
    python scripts/crear_fixture.py
"""

import pandas as pd

from src.config import LINEA_OBJETIVO_CODIGO, RUTA_FIXTURE, ruta_base
from src.datos.esquema import validar_base

DEPARTAMENTOS_FIXTURE = ["Antioquia", "Bogota"]
LINEAS_FIXTURE = [4, LINEA_OBJETIVO_CODIGO]
FILAS_ESPERADAS = 2 * 2 * 91  # departamentos × líneas × meses = 364


def crear_fixture() -> pd.DataFrame:
    """Lee la base central, filtra el recorte y lo valida contra el contrato."""
    base = pd.read_csv(ruta_base("central"), parse_dates=["fecha"])
    mini = base[
        base["departamento"].isin(DEPARTAMENTOS_FIXTURE)
        & base["linea_codigo"].isin(LINEAS_FIXTURE)
    ].reset_index(drop=True)

    if len(mini) != FILAS_ESPERADAS:
        raise ValueError(f"El fixture tiene {len(mini)} filas y se esperaban {FILAS_ESPERADAS}")
    validar_base(mini)  # mismo contrato que la base real
    return mini


def main() -> None:
    mini = crear_fixture()
    RUTA_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    # Mismo formato que la base original (4 decimales, fecha AAAA-MM-DD) para que el
    # archivo sea idéntico en cada corrida y git no marque cambios falsos
    mini.to_csv(RUTA_FIXTURE, index=False, float_format="%.4f", date_format="%Y-%m-%d")
    print(f"{RUTA_FIXTURE.name}: {len(mini)} filas, {mini['fecha'].nunique()} meses")


if __name__ == "__main__":
    main()

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    COLUMNAS_PROHIBIDAS,
    DEPARTAMENTOS,
    INICIO_PRUEBA,
    LINEAS,
    OBJETIVO,
    PREDICTORES_BIN,
    PREDICTORES_NUM,
    PREDICTORES_REZ1,
)
from src.datos.cargar import cargar_entrenamiento
from src.datos.esquema import COLUMNAS_BASE, validar_base
from src.evaluacion.particion import CVTemporalPorMes

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]
SCRIPT_EVALUACION_PRUEBA = Path("scripts/evaluar_prueba.py")


def _base_sintetica_valida() -> pd.DataFrame:
    fila = {columna: 1.0 for columna in COLUMNAS_BASE}
    fila.update(
        {
            "fecha": pd.Timestamp("2020-01-01"),
            "anio": 2020,
            "mes": 1,
            "departamento": DEPARTAMENTOS[0],
            "linea_codigo": 4,
            "linea": LINEAS[4],
            "es_muebles": 0,
            OBJETIVO: 100.0,
            "covid": 0,
        }
    )
    for columna in (*PREDICTORES_NUM, *PREDICTORES_REZ1, *PREDICTORES_BIN):
        fila[columna] = 1.0
    fila["covid"] = 0
    return pd.DataFrame([fila], columns=COLUMNAS_BASE)


@pytest.mark.parametrize("columna", COLUMNAS_PROHIBIDAS)
def test_validar_base_rechaza_todas_las_columnas_de_auditoria(columna):
    base = _base_sintetica_valida()
    base[columna] = 1.0

    with pytest.raises(ValueError, match="fuga"):
        validar_base(base)


def test_cargar_entrenamiento_excluye_fechas_desde_inicio_prueba(monkeypatch):
    datos_sinteticos = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                ["2025-07-01", "2025-08-01", "2026-07-01"]
            )
        }
    )
    monkeypatch.setattr(
        "src.datos.cargar._leer",
        lambda escenario, fixture: datos_sinteticos.copy(),
    )

    entrenamiento = cargar_entrenamiento("central")

    assert not entrenamiento.empty
    assert (entrenamiento["fecha"] < pd.Timestamp(INICIO_PRUEBA)).all()
    assert entrenamiento["fecha"].tolist() == [pd.Timestamp("2025-07-01")]


def _llama_a_cargar_prueba(codigo: str) -> bool:
    arbol = ast.parse(codigo)
    nombres_funcion = {"cargar_prueba"}

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module in {
            "src.datos",
            "src.datos.cargar",
        }:
            nombres_funcion.update(
                alias.asname or alias.name
                for alias in nodo.names
                if alias.name == "cargar_prueba"
            )

    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        if isinstance(nodo.func, ast.Name) and nodo.func.id in nombres_funcion:
            return True
        if isinstance(nodo.func, ast.Attribute) and nodo.func.attr == "cargar_prueba":
            return True
    return False


def test_cargar_prueba_no_se_invoca_fuera_del_script_previsto():
    directorios_productivos = (RAIZ_REPOSITORIO / "src", RAIZ_REPOSITORIO / "scripts")
    llamadores_no_permitidos = []

    for directorio in directorios_productivos:
        for archivo in directorio.rglob("*.py"):
            ruta_relativa = archivo.relative_to(RAIZ_REPOSITORIO)
            if ruta_relativa == SCRIPT_EVALUACION_PRUEBA:
                continue
            codigo = archivo.read_text(encoding="utf-8")
            if _llama_a_cargar_prueba(codigo):
                llamadores_no_permitidos.append(ruta_relativa.as_posix())

    assert not llamadores_no_permitidos, (
        "cargar_prueba tiene llamadores productivos fuera de "
        f"{SCRIPT_EVALUACION_PRUEBA}: {llamadores_no_permitidos}"
    )


def test_cv_temporal_no_mezcla_meses_en_entrenamiento_y_validacion():
    fechas = pd.date_range("2019-02-01", periods=78, freq="MS")
    datos_sinteticos = pd.DataFrame({"fecha": fechas.repeat(2)})

    for indices_entrenamiento, indices_validacion in CVTemporalPorMes().split(
        datos_sinteticos
    ):
        entrenamiento = datos_sinteticos.iloc[indices_entrenamiento]["fecha"]
        validacion = datos_sinteticos.iloc[indices_validacion]["fecha"]

        assert set(entrenamiento).isdisjoint(set(validacion))
        assert entrenamiento.max() < validacion.min()
        assert (entrenamiento < pd.Timestamp(INICIO_PRUEBA)).all()
        assert (validacion < pd.Timestamp(INICIO_PRUEBA)).all()
        assert validacion.nunique() == 6

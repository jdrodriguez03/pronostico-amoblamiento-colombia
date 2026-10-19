from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEPARTAMENTOS,
    ESCENARIOS,
    INICIO_PRUEBA,
    LINEA_OBJETIVO,
    LINEAS,
    N_MESES,
    PESOS_DEPTO,
)
from src.datos import panel as modulo_panel
from src.datos.cargar import cargar_entrenamiento, cargar_prueba
from src.evaluacion.agregacion import agregar_nacional


@pytest.fixture
def emc_sintetico(monkeypatch):
    fechas = pd.date_range("2019-01-01", periods=N_MESES, freq="MS")
    nacional = pd.DataFrame(
        [
            (fecha, nombre, 90.0 + fecha.month + codigo / 10)
            for fecha in fechas
            for codigo, nombre in LINEAS.items()
        ],
        columns=["fecha", "linea", "L"],
    ).sample(frac=1, random_state=5)
    indices = pd.DataFrame(
        [
            (fecha, departamento, float(80 + di + fecha.month / 10))
            for fecha in fechas
            for di, departamento in enumerate(DEPARTAMENTOS)
        ],
        columns=["fecha", "departamento", "indice_depto"],
    ).sample(frac=1, random_state=9)
    from src.datos.emc import factor_regional

    monkeypatch.setattr(modulo_panel, "leer_lineas_nacional", lambda: nacional.copy())
    monkeypatch.setattr(modulo_panel, "leer_departamental", lambda: indices.copy())
    monkeypatch.setattr(
        modulo_panel,
        "factor_regional",
        lambda _: factor_regional(indices),
    )
    return nacional, indices


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_panel_completo_y_componentes_reconstruyen_indice(escenario, emc_sintetico):
    nacional, _ = emc_sintetico
    panel, componentes = modulo_panel._construir_panel_y_componentes(escenario)

    assert len(panel) == N_MESES * len(DEPARTAMENTOS) * len(LINEAS) == 5096
    assert not panel.isna().any().any()
    assert (panel["indice_ventas_real"] > 0).all()
    assert panel[["fecha", "departamento", "linea_codigo"]].drop_duplicates().shape[0] == len(panel)
    assert set(panel["fecha"]) == set(nacional["fecha"])
    assert set(panel["departamento"]) == set(DEPARTAMENTOS)
    assert set(panel["linea_codigo"]) == set(LINEAS)
    assert not {"L", "F", "eta"} & set(panel.columns)
    np.testing.assert_allclose(
        componentes["L"] * componentes["F"] * componentes["eta"],
        panel["indice_ventas_real"],
        rtol=0,
        atol=1e-12,
    )

    obtenido = agregar_nacional(panel, "indice_ventas_real", linea=LINEA_OBJETIVO)
    oficial = nacional[nacional["linea"] == LINEA_OBJETIVO].set_index("fecha")["L"]
    np.testing.assert_allclose(obtenido.sort_index(), oficial.sort_index(), rtol=0, atol=1e-6)


def test_agregar_nacional_pondera_y_filtra_linea():
    datos = pd.DataFrame(
        [
            (pd.Timestamp("2020-01-01"), departamento, codigo, nombre, float(codigo))
            for departamento in DEPARTAMENTOS
            for codigo, nombre in LINEAS.items()
        ],
        columns=["fecha", "departamento", "linea_codigo", "linea", "valor"],
    )
    resultado = agregar_nacional(datos, "valor", linea=8)
    assert resultado.iloc[0] == pytest.approx(8 * sum(PESOS_DEPTO.values()))
    assert agregar_nacional(datos, "valor", linea=LINEA_OBJETIVO).equals(resultado)


def test_cargadores_particionan_panel_sin_solapamiento(tmp_path, monkeypatch):
    fechas = pd.date_range("2019-01-01", periods=N_MESES, freq="MS")
    panel = pd.DataFrame({"fecha": fechas, "valor": np.arange(N_MESES)})
    monkeypatch.setattr("src.datos.cargar.RUTA_PROCESSED", tmp_path)
    panel.to_parquet(tmp_path / "panel_central.parquet", index=False)

    entrenamiento = cargar_entrenamiento("central")
    prueba = cargar_prueba("central")
    inicio = pd.Timestamp(INICIO_PRUEBA)
    assert entrenamiento["fecha"].max() < inicio
    assert prueba["fecha"].min() == inicio
    assert set(entrenamiento["fecha"]).isdisjoint(prueba["fecha"])
    assert len(prueba) == 12


def test_escenario_desconocido_se_rechaza():
    with pytest.raises(ValueError, match="Escenario inválido"):
        modulo_panel.construir_panel("otro")


def test_script_escribe_paneles_y_componentes_en_rutas_configuradas(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import scripts.construir_panel as script

    monkeypatch.setattr(script, "RUTA_PROCESSED", tmp_path)
    monkeypatch.setattr(
        script,
        "_construir_panel_y_componentes",
        lambda escenario: (pd.DataFrame({"escenario": [escenario]}), pd.DataFrame({"L": [1.0]})),
    )
    script.main()

    for escenario in ESCENARIOS:
        assert (tmp_path / f"panel_{escenario}.parquet").exists()
        assert (tmp_path / f"componentes_{escenario}.parquet").exists()

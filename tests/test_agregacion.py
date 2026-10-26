import numpy as np
import pandas as pd
import pytest

from src.config import ESCENARIOS, OBJETIVO, RUTA_LINEAS_NACIONALES, ruta_base
from src.evaluacion.agregacion import agregar_nacional


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_agregar_nacional_reproduce_l8_en_todos_los_escenarios(escenario):
    panel = pd.read_csv(ruta_base(escenario), parse_dates=["fecha"])
    nacional = pd.read_csv(RUTA_LINEAS_NACIONALES, parse_dates=["fecha"])
    esperado = nacional.set_index("fecha")["L_8"].sort_index()

    obtenido = agregar_nacional(panel, OBJETIVO).sort_index()
    assert obtenido.index.equals(esperado.index)

    error_relativo = np.abs((obtenido - esperado) / esperado)
    assert error_relativo.max() < 1e-6

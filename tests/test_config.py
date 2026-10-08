# Dueño: Aleja
import pandas as pd

from src import config


def test_pesos_suman_uno_y_cubren_los_departamentos():
    assert abs(sum(config.PESOS_DEPTO.values()) - 1) < 1e-3
    assert set(config.PESOS_DEPTO) == set(config.DEPARTAMENTOS)


def test_tamano_del_panel():
    assert len(config.DEPARTAMENTOS) == 7
    assert len(config.LINEAS) == 8
    meses = pd.date_range(config.FECHA_INICIO, config.FECHA_FIN, freq="MS")
    assert len(meses) == config.N_MESES == 91
    # 7 departamentos x 8 líneas x 91 meses = 5.096 registros
    assert len(meses) * len(config.DEPARTAMENTOS) * len(config.LINEAS) == 5096


def test_particion_entrenamiento_prueba():
    meses = pd.date_range(config.FECHA_INICIO, config.FECHA_FIN, freq="MS")
    inicio = pd.Timestamp(config.INICIO_PRUEBA)
    assert (meses < inicio).sum() == 79
    assert (meses >= inicio).sum() == 12


def test_validacion_cruzada_cabe_en_entrenamiento():
    # El fold 1 debe entrenar con al menos 4 años
    entrenamiento = 79
    primer_fold = entrenamiento - config.CV_FOLDS * config.CV_MESES_VALIDACION
    assert primer_fold == 49


def test_columnas_coherentes():
    assert config.LINEA_OBJETIVO in config.LINEAS
    assert set(config.CIUDAD_ICC) <= set(config.DEPARTAMENTOS)
    usadas = {config.OBJETIVO, *config.PREDICTORES_NUM, *config.PREDICTORES_CAT,
              *config.PREDICTORES_BIN}
    assert not usadas & set(config.COLUMNAS_PROHIBIDAS)


def test_escenarios_fijos():
    assert config.ESCENARIOS == {"bajo": 0.02, "central": 0.07, "alto": 0.13}
    assert len(set(config.SEMILLAS_ESCENARIO.values())) == 3
    assert config.PHI == 0.6

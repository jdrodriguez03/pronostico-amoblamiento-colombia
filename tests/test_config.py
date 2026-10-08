# Dueño: Aleja
import pandas as pd

from src import config


def test_pesos_suman_uno_y_coinciden_con_auditoria():
    assert abs(sum(config.PESOS_DEPTO.values()) - 1) < 1e-5
    assert set(config.PESOS_DEPTO) == set(config.DEPARTAMENTOS)
    comp = pd.read_csv(config.RUTA_COMPONENTES)
    pesos = comp.groupby("departamento").peso_base2019.first().to_dict()
    for d, w in config.PESOS_DEPTO.items():
        assert abs(pesos[d] - w) < 1e-9


def test_tamano_del_panel():
    assert len(config.DEPARTAMENTOS) == 7
    assert len(config.LINEAS) == 8
    meses = pd.date_range(config.FECHA_INICIO, config.FECHA_FIN, freq="MS")
    assert len(meses) == config.N_MESES == 91
    assert len(meses) * len(config.DEPARTAMENTOS) * len(config.LINEAS) == 5096


def test_particion_entrenamiento_prueba():
    meses = pd.date_range(config.FECHA_INICIO, config.FECHA_FIN, freq="MS")
    inicio = pd.Timestamp(config.INICIO_PRUEBA)
    assert (meses < inicio).sum() == 79  # 78 si se descarta enero de 2019 (USAR_REZAGO)
    assert (meses >= inicio).sum() == 12


def test_columnas_coherentes():
    assert config.LINEA_OBJETIVO == "Electrodomesticos y muebles"
    usadas = {config.OBJETIVO, *config.PREDICTORES_NUM, *config.PREDICTORES_REZ1,
              *config.PREDICTORES_CAT, *config.PREDICTORES_BIN}
    assert not usadas & set(config.COLUMNAS_PROHIBIDAS)
    assert config.predictores_modelo() == (
        config.PREDICTORES_REZ1 if config.USAR_REZAGO else config.PREDICTORES_NUM
    )


def test_escenarios_fijos():
    assert config.ESCENARIOS == {"bajo": 0.02, "central": 0.07, "alto": 0.13}
    assert config.SEMILLA == 42
    assert config.RHO == 0.6

# Dueño: Aleja
"""Hoja de ajustes del proyecto: rutas y constantes que usan dos o más personas.

Regla 4 del repositorio: nadie escribe rutas ni números compartidos a mano;
se importan desde aquí, por ejemplo ``from src.config import SEMILLA, OBJETIVO``.
Si necesitas cambiar un valor, pídeselo a Aleja: un cambio aquí afecta a los tres.

Los valores coinciden con la base construida en la Entrega 1
(src/simulation/simular_base.py y src/features/predictores.py).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas (se arman desde la carpeta del repo: funcionan igual en cualquier PC y en el CI)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]

RUTA_DATA = RAIZ / "data"
RUTA_RAW = RUTA_DATA / "raw"
RUTA_INTERIM = RUTA_DATA / "interim"
RUTA_PROCESSED = RUTA_DATA / "processed"
RUTA_FIXTURES = RUTA_DATA / "fixtures"

RUTA_RESULTADOS = RAIZ / "results"
RUTA_TABLAS = RUTA_RESULTADOS / "tablas"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"  # en .gitignore: se regenera

RUTA_DOCS = RAIZ / "docs"

# Archivos de auditoría: sirven para verificar y para agregar a nivel nacional,
# pero NUNCA entran al modelo como predictores (serían fuga de información)
RUTA_COMPONENTES = RUTA_INTERIM / "componentes_regionales.csv"
RUTA_LINEAS_NACIONALES = RUTA_INTERIM / "lineas_nacionales.csv"
RUTA_FIXTURE = RUTA_FIXTURES / "base_mini.csv"


def ruta_base(escenario: str = "central") -> Path:
    """Base final (objetivo + predictores) de un escenario: la que entra a los modelos."""
    return RUTA_PROCESSED / f"base_final_{escenario}.csv"


# ---------------------------------------------------------------------------
# Reproducibilidad y simulación (fijados en la Entrega 1, antes de modelar)
# ---------------------------------------------------------------------------
SEMILLA = 42
# Desviación de eta en cada escenario; el central es el principal
ESCENARIOS = {"bajo": 0.02, "central": 0.07, "alto": 0.13}
ESCENARIO_PRINCIPAL = "central"
# Persistencia AR(1) de eta (RHO en simular_base.py)
RHO = 0.6

# ---------------------------------------------------------------------------
# Estructura del panel: 7 dominios × 8 líneas × 91 meses = 5.096 registros
# ---------------------------------------------------------------------------
# Tal como aparecen en la base (sin tildes)
DEPARTAMENTOS = [
    "Antioquia",
    "Atlantico",
    "Bogota",
    "Cundinamarca",
    "Santander",
    "Valle del Cauca",
    "Otros departamentos",
]

# Pesos base 2019 (columna peso_base2019 de componentes_regionales.csv)
PESOS_DEPTO = {
    "Antioquia": 0.170438,
    "Atlantico": 0.057136,
    "Bogota": 0.225540,
    "Cundinamarca": 0.049960,
    "Santander": 0.043320,
    "Valle del Cauca": 0.120603,
    "Otros departamentos": 0.333003,
}

# Código EMC -> nombre de la línea, tal como aparecen en la base
LINEAS = {
    4: "Prendas de vestir y textiles",
    8: "Electrodomesticos y muebles",
    9: "Articulos y utensilios domesticos",
    10: "Aseo del hogar",
    11: "Informatica y telecomunicaciones",
    12: "Sonido y video",
    14: "Ferreteria, vidrios y pinturas",
    15: "Otras mercancias de uso domestico",
}
# La línea sobre la que se evalúa el objetivo del artículo (MAPE nacional)
LINEA_OBJETIVO_CODIGO = 8
LINEA_OBJETIVO = LINEAS[LINEA_OBJETIVO_CODIGO]

# ---------------------------------------------------------------------------
# Fechas (primer día de cada mes)
# ---------------------------------------------------------------------------
FECHA_INICIO = "2019-01-01"
FECHA_FIN = "2026-07-01"
N_MESES = 91
# Primer mes del conjunto de prueba: agosto de 2025 a julio de 2026 (12 meses)
INICIO_PRUEBA = "2025-08-01"

# ---------------------------------------------------------------------------
# Columnas de base_final_*.csv (el contrato completo está en src/datos/esquema.py)
# ---------------------------------------------------------------------------
OBJETIVO = "indice_ventas_real"
# Predictores del mismo mes
PREDICTORES_NUM = [
    "icc",
    "inflacion_anual",
    "tasa_consumo",
    "importaciones_muebles_usd_miles",
    "area_vivienda_m2",
]
# Los mismos, con un mes de rezago (el dato del mes t no se conoce al pronosticar t)
PREDICTORES_REZ1 = [f"{c}_rez1" for c in PREDICTORES_NUM]
PREDICTORES_CAT = ["mes", "departamento", "linea"]
PREDICTORES_BIN = ["covid"]
# Columnas que identifican la fila; no son predictores
IDENTIFICADORES = ["fecha", "anio", "linea_codigo", "es_muebles"]

# Decisión del arranque (día 1): usar los predictores rezagados.
# Si es True, los modelos usan PREDICTORES_REZ1 y se descarta enero de 2019,
# el único mes sin rezago de vivienda (no se imputa).
USAR_REZAGO = True

# Columnas de los archivos de auditoría: NUNCA pueden ser predictores
COLUMNAS_PROHIBIDAS = [
    "peso_base2019",
    "factor_regional",
    "indice_grupo_depto",
    "indice_grupo_nacional",
    *[f"L_{c}" for c in LINEAS],
]

# ---------------------------------------------------------------------------
# Validación y modelos
# ---------------------------------------------------------------------------
CV_FOLDS = 5
CV_MESES_VALIDACION = 6
# Modelar log(indice) con TransformedTargetRegressor (decisión del arranque)
LOG_OBJETIVO = True
# Nivel de significancia del stepwise (el del artículo)
ALFA_STEPWISE = 0.25


def predictores_modelo() -> list[str]:
    """Columnas numéricas que entran al modelo según USAR_REZAGO."""
    return PREDICTORES_REZ1 if USAR_REZAGO else PREDICTORES_NUM

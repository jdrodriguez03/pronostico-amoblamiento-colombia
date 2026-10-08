# Dueño: Aleja
"""Hoja de ajustes del proyecto: rutas y constantes que usan dos o más personas.

Regla 4 del repositorio: nadie escribe rutas ni números compartidos a mano;
se importan desde aquí, por ejemplo ``from src.config import SEMILLA, ESCENARIOS``.
Si necesitas cambiar un valor, pídeselo a Aleja: un cambio aquí afecta a los tres.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas (se arman desde la carpeta del repo: funcionan igual en cualquier PC y en el CI)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]

RUTA_DATA = RAIZ / "data"
RUTA_RAW = RUTA_DATA / "raw"
RUTA_PROCESSED = RUTA_DATA / "processed"
RUTA_FIXTURES = RUTA_DATA / "fixtures"

RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_TABLAS = RUTA_RESULTADOS / "tablas"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"  # en .gitignore: se regenera

RUTA_DOCS = RAIZ / "docs"

RUTA_PREDICTORES = RUTA_PROCESSED / "predictores.parquet"
RUTA_FIXTURE_DATASET = RUTA_FIXTURES / "dataset_mini.parquet"


def ruta_panel(escenario: str) -> Path:
    """Panel con la variable objetivo (sin predictores) de un escenario."""
    return RUTA_PROCESSED / f"panel_{escenario}.parquet"


def ruta_componentes(escenario: str) -> Path:
    """L, F y eta de un escenario. Solo para verificar la simulación, nunca para modelar."""
    return RUTA_PROCESSED / f"componentes_{escenario}.parquet"


def ruta_dataset(escenario: str) -> Path:
    """Dataset final (panel + predictores) que entra a los modelos."""
    return RUTA_PROCESSED / f"dataset_{escenario}.parquet"


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEMILLA = 42

# ---------------------------------------------------------------------------
# Simulación del componente eta (fijado ANTES de modelar; ver Entrega 1, sección 6.3)
# ---------------------------------------------------------------------------
# Desviación estándar de log(eta) en cada escenario
ESCENARIOS = {"bajo": 0.02, "central": 0.07, "alto": 0.13}
# Una semilla distinta por escenario, derivada de SEMILLA
SEMILLAS_ESCENARIO = {nombre: SEMILLA + i for i, nombre in enumerate(ESCENARIOS)}
# Coeficiente del proceso AR(1) de eta
PHI = 0.6

# ---------------------------------------------------------------------------
# Estructura del panel
# ---------------------------------------------------------------------------
# Los 7 dominios departamentales de la EMC.
# Aleja: ajustar los textos para que coincidan EXACTAMENTE con los del anexo del DANE.
DEPARTAMENTOS = [
    "Antioquia",
    "Atlántico",
    "Bogotá",
    "Cundinamarca",
    "Santander",
    "Valle del Cauca",
    "Otros departamentos",
]

# Pesos departamentales en base 2019 (Entrega 1, sección 6.3)
PESOS_DEPTO = {
    "Otros departamentos": 0.333,
    "Bogotá": 0.226,
    "Antioquia": 0.170,
    "Valle del Cauca": 0.121,
    "Atlántico": 0.057,
    "Cundinamarca": 0.050,
    "Santander": 0.043,
}

# Las 8 líneas de mercancía del hogar de la EMC (anexo nacional, hoja 2.2).
# Aleja: ajustar los textos para que coincidan EXACTAMENTE con los del anexo del DANE.
LINEAS = [
    "Electrodomésticos, muebles para el hogar",
    "Artículos y utensilios domésticos",
    "Productos de aseo del hogar",
    "Equipo de informática y telecomunicaciones",
    "Equipo de sonido y video",
    "Ferretería, vidrios y pinturas",
    "Otras mercancías de uso doméstico",
    "Prendas de vestir y textiles",
]

# La línea sobre la que se evalúa el objetivo del artículo (MAPE nacional)
LINEA_OBJETIVO = LINEAS[0]

# ICC regional de Fedesarrollo: ciudad que representa a cada dominio.
# Los dominios que no están aquí usan el ICC nacional.
CIUDAD_ICC = {
    "Bogotá": "Bogotá",
    "Antioquia": "Medellín",
    "Valle del Cauca": "Cali",
    "Atlántico": "Barranquilla",
    "Santander": "Bucaramanga",
}

# ---------------------------------------------------------------------------
# Fechas (primer día de cada mes)
# ---------------------------------------------------------------------------
FECHA_INICIO = "2019-01-01"
FECHA_FIN = "2026-07-01"
N_MESES = 91
# Primer mes del conjunto de prueba: agosto de 2025 a julio de 2026 (12 meses)
INICIO_PRUEBA = "2025-08-01"

# Indicadora de pandemia (a confirmar por Julián el día 3 mirando las series)
PANDEMIA_INICIO = "2020-03-01"
PANDEMIA_FIN = "2021-06-01"

# ---------------------------------------------------------------------------
# Columnas del dataset (el contrato completo está en src/datos/esquema.py)
# ---------------------------------------------------------------------------
OBJETIVO = "indice"
PREDICTORES_NUM = ["icc", "area_aprobada", "tasa_consumo", "inflacion", "importaciones"]
PREDICTORES_CAT = ["mes", "departamento", "linea"]
PREDICTORES_BIN = ["pandemia"]
# Componentes de la construcción del índice: NUNCA pueden ser predictores (regla de oro 4)
COLUMNAS_PROHIBIDAS = ["L", "F", "eta"]

# ---------------------------------------------------------------------------
# Validación y modelos
# ---------------------------------------------------------------------------
CV_FOLDS = 5
CV_MESES_VALIDACION = 6
# Modelar log(indice) con TransformedTargetRegressor (decisión del arranque)
LOG_OBJETIVO = True
# Nivel de significancia del stepwise (el del artículo)
ALFA_STEPWISE = 0.25

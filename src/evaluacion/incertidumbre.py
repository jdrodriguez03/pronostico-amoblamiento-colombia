# Dueño: Juanes
"""Incertidumbre del error con bootstrap por bloques de meses. Entrega final, lunes 9.

Responde a la corrección 5 de la profesora: el MAPE de prueba sale de solo 12 meses, así
que hay que decir qué tan seguro es ese número. Se usa un bootstrap, pero NUNCA fila por
fila (regla 5 de la entrega):

- Las 56 filas de un mes (7 departamentos × 8 líneas) comparten los predictores nacionales:
  no son 56 observaciones independientes. Por eso el bootstrap trabaja sobre la serie
  NACIONAL mensual (un valor por mes), no sobre las filas del panel.
- Los meses seguidos tampoco son independientes: si el modelo se queda corto en octubre,
  probablemente también en noviembre. Un bootstrap que sorteara meses sueltos rompería esa
  dependencia y daría un intervalo demasiado angosto. Por eso se sortean BLOQUES de meses
  consecutivos (bootstrap de bloques móviles, Künsch 1989).

Cómo funciona, con el ejemplo de la prueba (n = 12 meses, bloques de 3):

    meses:            ago sep oct nov dic ene feb mar abr may jun jul
    posición:          0   1   2   3   4   5   6   7   8   9  10  11
    bloques posibles: (0,1,2) (1,2,3) (2,3,4) ... (9,10,11)   → 10 bloques "móviles"
    una réplica:      se sortean 4 bloques con reemplazo (4 × 3 = 12 meses), por ejemplo
                      (5,6,7) (0,1,2) (5,6,7) (8,9,10) → se calcula el MAPE de esos 12 meses
    B = 2000 réplicas → 2000 MAPE → el IC 95 % son sus percentiles 2,5 y 97,5.

Dentro de cada bloque los meses quedan en su orden y seguidos, así que la dependencia de
corto plazo se conserva. Si n no es múltiplo del largo del bloque, se sortean bloques de más
y se recorta a n.

Funciones:
    indices_bloques(n, ...)            las posiciones sorteadas (para poder probarlas)
    bootstrap_bloques(y_real, y_pred)  MAPE con su IC 95 %
    bootstrap_diferencia(y_real, a, b) diferencia de MAPE entre dos modelos, PAREADA: los dos
                                       se miden en las mismas réplicas, como en tabla_pareada
    rango_error(df_oof)                percentiles 10, 50 y 90 del error relativo fuera de
                                       muestra, para el rango p10–p90 de la app (rango_error.csv)
"""

import math

import numpy as np
import pandas as pd

from src.config import SEMILLA
from src.evaluacion.agregacion import agregar_nacional
from src.evaluacion.metricas import mape

LARGO_BLOQUE = 3   # meses consecutivos por bloque (un trimestre)
B_REPLICAS = 2000  # réplicas bootstrap
NIVEL = 0.95       # nivel del intervalo
CUANTILES_RANGO = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
COLUMNAS_RANGO = ["ambito", "linea_codigo", *CUANTILES_RANGO]
NACIONAL = "Nacional"


# ---------------------------------------------------------------------------
# Sorteo de los bloques
# ---------------------------------------------------------------------------
def indices_bloques(n: int, largo_bloque: int = LARGO_BLOQUE, B: int = B_REPLICAS,
                    semilla: int = SEMILLA) -> np.ndarray:
    """Matriz B × n de posiciones: cada fila es una réplica armada con bloques seguidos.

    Ejemplo con n = 6 y largo 3: los bloques posibles empiezan en 0, 1, 2 o 3; una fila
    puede ser [2, 3, 4, 0, 1, 2] (bloque que empieza en 2 y bloque que empieza en 0).
    La misma semilla da siempre la misma matriz (reproducibilidad).
    """
    if largo_bloque < 1 or largo_bloque > n:
        raise ValueError(f"largo_bloque debe estar entre 1 y n = {n}, no {largo_bloque}")
    if B < 1:
        raise ValueError("B debe ser al menos 1")
    rng = np.random.default_rng(semilla)
    n_bloques = math.ceil(n / largo_bloque)
    # Dónde empieza cada bloque: de 0 a n − largo (el último bloque que cabe completo)
    inicios = rng.integers(0, n - largo_bloque + 1, size=(B, n_bloques))
    # inicio + (0, 1, ..., largo − 1) → los meses seguidos de cada bloque
    posiciones = inicios[:, :, None] + np.arange(largo_bloque)
    return posiciones.reshape(B, -1)[:, :n]


def _serie_mensual(y_real, *predicciones) -> list[np.ndarray]:
    """Arreglos alineados y en orden de fecha; rechaza todo lo que no sea una serie mensual.

    Si llegan pd.Series con fechas como índice, se ordenan por fecha y se exige un solo valor
    por mes: así nadie le pasa por error las filas del panel (habría meses repetidos).
    """
    if isinstance(y_real, pd.Series):
        fechas = y_real.index
        if fechas.has_duplicates:
            raise ValueError("hay meses repetidos: el bootstrap es sobre la serie nacional "
                             "(un valor por mes), nunca sobre las filas del panel")
        orden = np.argsort(np.asarray(fechas))
        series = [y_real, *predicciones]
        for s in predicciones:
            if isinstance(s, pd.Series) and not s.index.equals(fechas):
                raise ValueError("observado y predicho deben tener las mismas fechas")
        arreglos = [np.asarray(s, dtype=float)[orden] for s in series]
    else:
        arreglos = [np.asarray(s, dtype=float).ravel() for s in (y_real, *predicciones)]
    n = arreglos[0].size
    if any(a.size != n for a in arreglos):
        raise ValueError("observado y predicho deben tener el mismo número de meses")
    if np.isnan(np.concatenate(arreglos)).any():
        raise ValueError("hay valores vacíos (NaN)")
    if (arreglos[0] == 0).any():
        raise ValueError("el MAPE no está definido cuando algún valor observado es 0")
    return arreglos


def _ape(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Error porcentual absoluto de cada mes: el MAPE es su promedio."""
    return np.abs(y - yhat) / np.abs(y) * 100


def _intervalo(valores: np.ndarray, nivel: float) -> tuple[float, float]:
    cola = (1 - nivel) / 2 * 100
    inf, sup = np.percentile(valores, [cola, 100 - cola])
    return float(inf), float(sup)


# ---------------------------------------------------------------------------
# MAPE con intervalo
# ---------------------------------------------------------------------------
def bootstrap_bloques(y_real, y_pred, largo_bloque: int = LARGO_BLOQUE,
                      B: int = B_REPLICAS, semilla: int = SEMILLA,
                      nivel: float = NIVEL) -> dict:
    """MAPE nacional y su IC por percentiles con bootstrap de bloques móviles.

    y_real, y_pred: la serie nacional mensual (por ejemplo, la línea 8 agregada con
        agregar_nacional), como pd.Series indexadas por fecha o como arreglos ya ordenados.

    Devuelve {"mape", "ic_inf", "ic_sup", "desv_boot", "n_meses", "largo_bloque", "B"}.
    "mape" es la estimación puntual (con los meses originales, sin remuestrear); el IC dice
    entre qué valores se movería el MAPE si los 12 meses de prueba hubieran sido otros
    parecidos. Con 12 meses el intervalo sale ancho, y eso se declara, no se esconde.

    Ejemplo: {"mape": 7.8, "ic_inf": 5.1, "ic_sup": 10.9, ...} se lee "MAPE de prueba de
    7,8 % (IC 95 % por bloques: 5,1 % a 10,9 %)".
    """
    y, yhat = _serie_mensual(y_real, y_pred)
    ape = _ape(y, yhat)
    idx = indices_bloques(y.size, largo_bloque, B, semilla)
    replicas = ape[idx].mean(axis=1)  # el MAPE de cada réplica
    inf, sup = _intervalo(replicas, nivel)
    return {
        "mape": mape(y, yhat),
        "ic_inf": inf,
        "ic_sup": sup,
        "desv_boot": float(replicas.std(ddof=1)),
        "n_meses": int(y.size),
        "largo_bloque": largo_bloque,
        "B": B,
    }


def bootstrap_diferencia(y_real, pred_a, pred_b, largo_bloque: int = LARGO_BLOQUE,
                         B: int = B_REPLICAS, semilla: int = SEMILLA,
                         nivel: float = NIVEL) -> dict:
    """Diferencia MAPE(a) − MAPE(b) con IC, usando LAS MISMAS réplicas para los dos modelos.

    Pareado quiere decir: en cada réplica se toman los mismos meses para a y para b. Así un
    mes difícil para los dos pesa igual en los dos y se cancela al restar, que es lo mismo
    que hace tabla_pareada con los folds. Si el IC no contiene el 0, la diferencia es
    distinguible del azar de los 12 meses; si lo contiene, los dos modelos empatan en la
    práctica.

    Devuelve {"dif", "ic_inf", "ic_sup", "prop_a_mejor", "n_meses", "largo_bloque", "B"};
    prop_a_mejor es la fracción de réplicas en que a tiene menor MAPE que b.

    Ejemplo (a = Ridge, b = KNN): dif = −0,4 con IC (−1,9; 1,2) → Ridge se equivoca 0,4
    puntos menos, pero el IC cruza el 0: empate práctico.
    """
    y, ya, yb = _serie_mensual(y_real, pred_a, pred_b)
    ape_a, ape_b = _ape(y, ya), _ape(y, yb)
    idx = indices_bloques(y.size, largo_bloque, B, semilla)
    replicas = ape_a[idx].mean(axis=1) - ape_b[idx].mean(axis=1)
    inf, sup = _intervalo(replicas, nivel)
    return {
        "dif": mape(y, ya) - mape(y, yb),
        "ic_inf": inf,
        "ic_sup": sup,
        "prop_a_mejor": float((replicas < 0).mean()),
        "n_meses": int(y.size),
        "largo_bloque": largo_bloque,
        "B": B,
    }


# ---------------------------------------------------------------------------
# Rango de error para la app (rango_error.csv)
# ---------------------------------------------------------------------------
def rango_error(df_oof: pd.DataFrame) -> pd.DataFrame:
    """Percentiles 10, 50 y 90 del error relativo fuera de muestra, por ámbito y línea.

    df_oof: predicciones FUERA DE MUESTRA del modelo final, una fila por mes, departamento y
        línea, con las columnas fecha, departamento, linea_codigo, y_real, y_pred. Salen de
        los meses de validación de los 5 folds (cada mes predicho por un modelo que no lo
        vio), nunca de la prueba. Debe traer un solo modelo.

    Error relativo de cada mes: e = (y_real − y_pred) / y_pred. Positivo = el modelo se quedó
    corto (mismo signo que el residuo de Aleja). Se divide por la PREDICCIÓN y no por el
    observado porque la app solo conoce la predicción: el rango que muestra es

        [ŷ · (1 + p10),  ŷ · (1 + p90)]

    Ejemplo: ŷ = 120, p10 = −0,05 y p90 = 0,08 → «entre 114 y 129,6»: en 8 de cada 10 meses
    de validación el valor observado quedó dentro de un rango así de ancho.

    ambito es el departamento o "Nacional"; para Nacional se agregan observado y predicho de
    cada línea con agregar_nacional (pesos PESOS_DEPTO) antes de calcular el error, igual que
    hace la app cuando se elige "Nacional (agregado)".
    """
    necesarias = ["fecha", "departamento", "linea_codigo", "y_real", "y_pred"]
    faltan = [c for c in necesarias if c not in df_oof.columns]
    if faltan:
        raise ValueError(f"a df_oof le faltan las columnas {faltan}")
    if "modelo" in df_oof.columns and df_oof["modelo"].nunique() > 1:
        raise ValueError("rango_error es de un solo modelo (el final): filtra df_oof antes")
    if df_oof.duplicated(["fecha", "departamento", "linea_codigo"]).any():
        raise ValueError("hay filas repetidas por fecha, departamento y línea")

    def _cuantiles(errores: pd.Series) -> dict:
        return {nombre: float(errores.quantile(q)) for nombre, q in CUANTILES_RANGO.items()}

    filas = []
    # Por departamento y línea
    datos = df_oof[necesarias].copy()
    datos["error"] = (datos["y_real"] - datos["y_pred"]) / datos["y_pred"]
    for (depto, linea), grupo in datos.groupby(["departamento", "linea_codigo"], sort=True):
        filas.append({"ambito": depto, "linea_codigo": int(linea), **_cuantiles(grupo["error"])})
    # Nacional: agregar primero, luego el error (el error nacional no es el promedio de errores)
    for linea in sorted(datos["linea_codigo"].unique()):
        obs = agregar_nacional(datos, "y_real", int(linea))
        pred = agregar_nacional(datos, "y_pred", int(linea))
        filas.append({"ambito": NACIONAL, "linea_codigo": int(linea),
                      **_cuantiles((obs - pred) / pred)})
    return pd.DataFrame(filas, columns=COLUMNAS_RANGO)

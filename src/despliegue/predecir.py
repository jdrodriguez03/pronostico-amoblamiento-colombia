# Dueño: Aleja
"""Lógica de la app: qué datos recibe, con qué modelo predice y qué devuelve.

Se separa de app/app.py para poder probarla sin Streamlit (tests/test_predecir.py).

Coherencia entre entrenamiento y uso: la app usa EXACTAMENTE la misma preparación que el
entrenamiento (agregar_rezagos_reales de Julián, mismas columnas predictoras) y el mismo
Pipeline (one-hot, logaritmo de vivienda, escalado y modelo van dentro). Por eso a predecir()
se le pasan filas "crudas" de la base: el Pipeline las transforma igual que en el ajuste.

Excepción documentada a la regla de la prueba: cargar_insumos_app() lee la base completa
(hasta julio de 2026) porque la app necesita los valores de los predictores de cualquier mes
para llenar las entradas. La app NO evalúa: no calcula métricas con esos meses ni elige nada
con ellos. El modelo se entrena solo con febrero 2019 – julio 2025 (cargar_entrenamiento).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import FECHA_FIN, INICIO_PRUEBA, LINEAS, OBJETIVO, RUTA_TABLAS, ruta_base
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.agregacion import agregar_nacional
from src.features.rezagos import REZAGOS_REALES, agregar_rezagos_reales, columnas_predictoras
from src.modelos.pipeline import ORDEN_FILAS, preparar_datos

NACIONAL = "Nacional (agregado)"
RUTA_RANGO = RUTA_TABLAS / "rango_error.csv"

NOMBRES_PREDICTORES = {
    "icc": "Índice de confianza del consumidor (Fedesarrollo)",
    "inflacion_anual": "Inflación anual, % (DANE, IPC)",
    "tasa_consumo": "Tasa de interés de consumo, % (Superfinanciera)",
    "importaciones_muebles_usd_miles": "Importaciones de muebles, USD miles (DANE-DIAN)",
    "area_vivienda_m2": "Área licenciada para vivienda, m² (DANE, ELIC)",
}


def _agregar_mes_siguiente(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega filas del mes siguiente al último de la base, sin objetivo ni predictores del mes.

    Como todos los rezagos son >= 1, sus predictores rezagados ya se conocen: es el único mes
    que se pronostica de verdad hacia adelante (ex ante). Los demás meses son ex post.
    """
    ultima = df["fecha"].max()
    nuevas = df[df["fecha"] == ultima].copy()
    siguiente = ultima + pd.DateOffset(months=1)
    nuevas["fecha"] = siguiente
    nuevas["anio"] = siguiente.year
    nuevas["mes"] = siguiente.month
    nuevas["covid"] = 0
    nuevas[OBJETIVO] = np.nan
    for var in REZAGOS_REALES:  # el dato del propio mes aún no existe
        nuevas[var] = np.nan
    return pd.concat([df, nuevas], ignore_index=True)


def cargar_insumos_app(escenario: str = "central") -> pd.DataFrame:
    """Base con los predictores rezagados de forma realista, lista para llenar las entradas.

    Lee base_final_{escenario}.csv con ruta_base, agrega el mes siguiente al último dato y
    aplica agregar_rezagos_reales (descarta los primeros meses sin dato rezagado).
    """
    df = pd.read_csv(ruta_base(escenario), parse_dates=["fecha"])
    df = _agregar_mes_siguiente(df)
    df = agregar_rezagos_reales(df, REZAGOS_REALES)
    return df.sort_values(ORDEN_FILAS, kind="stable").reset_index(drop=True)


def entrenar_para_app(escenario: str = "central", fixture: bool = False):
    """Modelo ajustado con febrero 2019 – julio 2025 (nunca con la prueba).

    Usa construir_modelo_final() de Juanes (src/modelos/final.py), que devuelve el modelo que
    eligió la regla escrita antes de abrir la prueba (KNN, k = 3). Si ese archivo todavía no está
    en la rama, arma el mismo KNN con los hiperparámetros congelados. Devuelve (modelo, texto).
    """
    datos, predictores = preparar_datos(cargar_entrenamiento(escenario, fixture=fixture), "reales")
    try:
        from src.modelos.final import MODELO_FINAL, construir_modelo_final

        modelo = construir_modelo_final()
        descripcion = f"{MODELO_FINAL.upper()}, modelo final según la regla previa a la prueba"
    except ImportError:
        from src.modelos.pipeline import candidato_con_predictores, prefijo_parametros

        modelo, _ = candidato_con_predictores("knn", predictores)
        pre = prefijo_parametros()
        modelo.set_params(**{f"{pre}n_neighbors": 3, f"{pre}weights": "uniform"})
        descripcion = "KNN, k = 3 (hiperparámetros congelados)"
    modelo.fit(datos, datos[OBJETIVO])
    return modelo, descripcion


def filas_a_predecir(df: pd.DataFrame, ambito: str, linea_codigo: int, fecha) -> pd.DataFrame:
    """Filas de la base para ese mes y línea: una si es departamento, siete si es Nacional."""
    fecha = pd.Timestamp(fecha)
    filas = df[(df["fecha"] == fecha) & (df["linea_codigo"] == linea_codigo)]
    if ambito != NACIONAL:
        filas = filas[filas["departamento"] == ambito]
    if filas.empty:
        raise ValueError(f"No hay datos para {ambito}, línea {linea_codigo}, {fecha:%Y-%m}")
    return filas.copy()


def predecir(modelo, entradas: pd.DataFrame) -> float:
    """Índice pronosticado (2019 = 100) para las filas de entrada.

    Una fila (departamento): su predicción. Varias filas de un mismo mes y línea (Nacional):
    se predice cada departamento y se agrega con agregar_nacional (pesos PESOS_DEPTO), igual
    que en la evaluación.
    """
    pred = np.asarray(modelo.predict(entradas), dtype=float)
    if len(entradas) == 1:
        return float(pred[0])
    datos = entradas[["fecha", "departamento", "linea_codigo"]].copy()
    datos["pred"] = pred
    linea = int(datos["linea_codigo"].iloc[0])
    return float(agregar_nacional(datos, "pred", linea).iloc[0])


def referencia_historica(df: pd.DataFrame, ambito: str, linea_codigo: int, fecha) -> float | None:
    """Valor OBSERVADO del mismo mes del año anterior (None si no existe)."""
    anterior = pd.Timestamp(fecha) - pd.DateOffset(years=1)
    try:
        filas = filas_a_predecir(df, ambito, linea_codigo, anterior)
    except ValueError:
        return None
    if filas[OBJETIVO].isna().any():
        return None
    if ambito != NACIONAL:
        return float(filas[OBJETIVO].iloc[0])
    return float(agregar_nacional(filas, OBJETIVO, linea_codigo).iloc[0])


def rango_prediccion(prediccion: float, ambito: str,
                     linea_codigo: int) -> tuple[float, float] | None:
    """[ŷ(1 + p10), ŷ(1 + p90)] desde rango_error.csv; None si el archivo no existe."""
    if not RUTA_RANGO.exists():
        return None
    tabla = pd.read_csv(RUTA_RANGO)
    clave = "Nacional" if ambito == NACIONAL else ambito
    fila = tabla[(tabla["ambito"] == clave) & (tabla["linea_codigo"] == linea_codigo)]
    if fila.empty:
        return None
    p10, p90 = float(fila["p10"].iloc[0]), float(fila["p90"].iloc[0])
    return prediccion * (1 + p10), prediccion * (1 + p90)


def tabla_entradas(filas: pd.DataFrame) -> pd.DataFrame:
    """Cada predictor con su valor y el mes al que corresponde según su rezago realista.

    Para Nacional los predictores numéricos son iguales en los 7 departamentos salvo el área
    de vivienda (departamental), que se muestra como la suma.
    """
    fecha = pd.Timestamp(filas["fecha"].iloc[0])
    columnas = columnas_predictoras()
    registros = []
    for var, col in zip(REZAGOS_REALES, columnas, strict=True):
        k = REZAGOS_REALES[var]
        es_suma = bool(var == "area_vivienda_m2" and len(filas) > 1)
        valor = filas[col].sum() if es_suma else filas[col].iloc[0]
        registros.append({
            "columna": col,
            "predictor": NOMBRES_PREDICTORES[var],
            "rezago (meses)": k,
            "dato del mes": f"{fecha - pd.DateOffset(months=k):%Y-%m}",
            "valor": float(valor),
            "es_suma": es_suma,
        })
    return pd.DataFrame(registros)


def aplicar_entradas(filas: pd.DataFrame, tabla: pd.DataFrame) -> pd.DataFrame:
    """Copia de filas con los valores (posiblemente editados) de tabla_entradas.

    En Nacional, un predictor nacional editado se copia a los 7 departamentos; el área de
    vivienda (suma) se reparte en proporción a lo que tenía cada departamento.
    """
    nuevas = filas.copy()
    for _, fila in tabla.iterrows():
        col, valor = fila["columna"], float(fila["valor"])
        if col.startswith("area_vivienda_m2") and len(nuevas) > 1:
            total = nuevas[col].sum()
            nuevas[col] = nuevas[col] * (valor / total) if total > 0 else valor / len(nuevas)
        else:
            nuevas[col] = valor
    return nuevas


def es_prueba_o_futuro(fecha) -> str:
    """Etiqueta del periodo del mes elegido, para que la app lo diga en pantalla."""
    fecha = pd.Timestamp(fecha)
    if fecha > pd.Timestamp(FECHA_FIN):
        return "ex ante (mes futuro: aún no hay dato observado)"
    if fecha >= pd.Timestamp(INICIO_PRUEBA):
        return "periodo de prueba (ex post)"
    return "periodo de entrenamiento (ex post, el modelo ya vio este mes)"


def nombre_linea(codigo: int) -> str:
    return LINEAS[codigo]


# ---------------------------------------------------------------------------
# Validación de entradas y monitoreo de drift (notebook de clase de MLOps, secciones 10,
# 16, 17 y 35): el modelo solo "conoce" el rango de los meses con que se entrenó.
# ---------------------------------------------------------------------------
ALFA_DRIFT = 0.05  # nivel del test KS, como en el notebook de clase


def rangos_entrenamiento(df: pd.DataFrame) -> pd.DataFrame:
    """Mínimo y máximo de cada columna predictora en los meses de entrenamiento."""
    ent = df[df["fecha"] < pd.Timestamp(INICIO_PRUEBA)]
    return ent[columnas_predictoras()].agg(["min", "max"]).T


def fuera_de_rango(tabla: pd.DataFrame, rangos: pd.DataFrame) -> list[str]:
    """Predictores de tabla_entradas cuyo valor cae fuera del rango de entrenamiento.

    En Nacional el área de vivienda se muestra como suma de departamentos, así que se compara
    contra el rango de la suma y no se marca aquí (se omite para no dar falsas alarmas).
    """
    fuera = []
    for _, fila in tabla.iterrows():
        col = fila["columna"]
        if col not in rangos.index or (col.startswith("area_vivienda_m2") and fila.get("es_suma")):
            continue
        if not rangos.loc[col, "min"] <= fila["valor"] <= rangos.loc[col, "max"]:
            fuera.append(fila["predictor"])
    return fuera


def _serie_mensual_predictores(df: pd.DataFrame) -> pd.DataFrame:
    """Un valor por mes de cada predictor sin rezago (vivienda: suma de departamentos)."""
    datos = df[(df["linea_codigo"] == df["linea_codigo"].min()) & df["icc"].notna()]
    agg = {v: "first" for v in REZAGOS_REALES}
    agg["area_vivienda_m2"] = "sum"
    return datos.groupby("fecha").agg(agg).sort_index()


def drift_predictores(df: pd.DataFrame) -> pd.DataFrame:
    """Compara los predictores que llegaron DESPUÉS del entrenamiento con los del entrenamiento.

    Referencia: meses de entrenamiento. Actual: meses desde INICIO_PRUEBA con dato publicado.
    Usa solo los predictores (X), nunca el objetivo: así se monitorea un modelo en producción.
    KS de dos muestras como en el notebook de clase. Ojo: los meses seguidos no son
    independientes, así que los p-valores son orientativos (tienden a ser optimistas).
    """
    serie = _serie_mensual_predictores(df)
    ref = serie[serie.index < pd.Timestamp(INICIO_PRUEBA)]
    act = serie[serie.index >= pd.Timestamp(INICIO_PRUEBA)]
    filas = []
    for var in serie.columns:
        estad, p = ks_2samp(ref[var], act[var])
        fuera = int(((act[var] < ref[var].min()) | (act[var] > ref[var].max())).sum())
        filas.append({
            "predictor": NOMBRES_PREDICTORES[var],
            "media entrenamiento": ref[var].mean(),
            "media reciente": act[var].mean(),
            "KS": estad,
            "p-valor": p,
            "drift": bool(p < ALFA_DRIFT),
            "meses fuera de rango": f"{fuera} de {len(act)}",
        })
    return pd.DataFrame(filas)


def semaforo_drift(tabla_drift: pd.DataFrame) -> str:
    """Semáforo del notebook de clase (sección 17) según el porcentaje de predictores con drift."""
    tasa = tabla_drift["drift"].mean()
    if tasa > 0.5:
        return f"ROJO: {tasa:.0%} de los predictores con drift; revisar el modelo antes de usarlo"
    if tasa > 0.2:
        return f"AMARILLO: {tasa:.0%} de los predictores con drift; monitorear de cerca"
    return f"VERDE: {tasa:.0%} de los predictores con drift"

# Dueño: Aleja
"""App de Streamlit: pronóstico del índice de ventas reales (2019 = 100).

Cómo correrla:  streamlit run app/app.py
"""

import sys
from pathlib import Path

# Para que "import src..." funcione aunque Streamlit se lance desde otra carpeta
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DEPARTAMENTOS, LINEA_OBJETIVO_CODIGO, LINEAS  # noqa: E402
from src.despliegue.predecir import (  # noqa: E402
    NACIONAL,
    aplicar_entradas,
    cargar_insumos_app,
    drift_predictores,
    entrenar_para_app,
    es_prueba_o_futuro,
    filas_a_predecir,
    fuera_de_rango,
    predecir,
    rango_prediccion,
    rangos_entrenamiento,
    referencia_historica,
    semaforo_drift,
    tabla_entradas,
)

st.set_page_config(page_title="Pronóstico índice de ventas de muebles", layout="centered")


@st.cache_data
def _insumos() -> pd.DataFrame:
    return cargar_insumos_app("central")


@st.cache_resource
def _modelo():
    # KNN y Ridge entrenan en segundos: no se sube ningún modelo al repositorio
    return entrenar_para_app("central")


# 1. Título y para qué sirve ---------------------------------------------------
st.title("Pronóstico del índice de ventas reales (2019 = 100)")
st.write(
    "Estima el índice de ventas reales del comercio minorista de bienes del hogar para un mes, "
    "a partir de indicadores económicos ya publicados el primer día hábil de ese mes. "
    "Sirve como **señal para la planeación de demanda** (dirección y magnitud del mes), "
    "no como una orden de compra."
)

df = _insumos()
modelo, descripcion_modelo = _modelo()

with st.sidebar:  # configuración, como en la app de clase
    st.header("Configuración")
    mostrar_entradas = st.checkbox("Mostrar y editar datos de entrada", value=True)
    mostrar_monitoreo = st.checkbox("Mostrar monitoreo de drift", value=False)

# 2. Selectores ----------------------------------------------------------------
c1, c2, c3 = st.columns(3)
ambito = c1.selectbox("Ámbito", [NACIONAL, *DEPARTAMENTOS])
codigos = list(LINEAS)
linea = c2.selectbox("Línea", codigos, index=codigos.index(LINEA_OBJETIVO_CODIGO),
                     format_func=lambda c: f"{c} · {LINEAS[c]}")
meses = sorted(df["fecha"].unique(), reverse=True)
fecha = pd.Timestamp(c3.selectbox("Mes a pronosticar", meses,
                                  format_func=lambda f: pd.Timestamp(f).strftime("%Y-%m")))
st.caption(f"Periodo: {es_prueba_o_futuro(fecha)} · Modelo: {descripcion_modelo}")

# 3. Datos de entrada ------------------------------------------------------------
filas = filas_a_predecir(df, ambito, linea, fecha)
entradas = tabla_entradas(filas)
if mostrar_entradas:
    st.subheader("Datos de entrada")
    st.write("Valores prellenados desde la base, con el mes al que corresponde cada dato según su "
             "rezago realista. Puede editar la columna «valor» para explorar otro escenario.")
    entradas = st.data_editor(
        entradas, hide_index=True, use_container_width=True,
        column_config={"es_suma": None},  # columna interna, no se muestra
        disabled=["columna", "predictor", "rezago (meses)", "dato del mes"],
    )
# Validación de rangos (buena práctica de despliegue del notebook de clase)
fuera = fuera_de_rango(entradas, rangos_entrenamiento(df))
if fuera:
    st.warning("Fuera del rango visto en el entrenamiento: " + ", ".join(fuera) +
               ". El modelo está extrapolando; tome el resultado con más cautela.")

# 4. Resultado -------------------------------------------------------------------
pred = predecir(modelo, aplicar_entradas(filas, entradas))
ref = referencia_historica(df, ambito, linea, fecha)
st.subheader("Resultado")
m1, m2, m3 = st.columns(3)
m1.metric("Índice pronosticado", f"{pred:,.1f}")
if ref is not None:
    m2.metric("Mismo mes del año anterior (observado)", f"{ref:,.1f}")
    m3.metric("Variación frente al año anterior", f"{100 * (pred / ref - 1):+.1f} %")
else:
    m2.metric("Mismo mes del año anterior", "no disponible")
rango = rango_prediccion(pred, ambito, linea)
if rango is None:
    st.write("Rango p10–p90: **rango no disponible**.")
else:
    st.write(f"Rango p10–p90: **entre {rango[0]:,.1f} y {rango[1]:,.1f}**. En 8 de cada 10 meses "
             "de validación el valor observado quedó dentro de un rango así de ancho.")

# 5. Limitaciones (fijo) -------------------------------------------------------------
st.info(
    "**Limitaciones.** El resultado es un **índice** de ventas reales (2019 = 100), no unidades, "
    "pesos ni inventario. Para decidir inventario hacen falta unidades, costos, inventario actual "
    "y tiempos de reposición. Los predictores son valores observados: salvo el último mes, el "
    "pronóstico es *ex post*. En la prueba (ago 2025 – jul 2026) subestimó los 12 meses "
    "(MAPE 19,8 %) y no superó al ingenuo estacional: en un mercado que crece, léalo como piso."
)
if ambito != NACIONAL:
    st.warning(
        "**Aviso regional.** El componente propio de cada departamento y línea es **simulado** "
        "(η, escenario central): esta cifra vale solo dentro de ese escenario y no es un dato "
        "observado del DANE."
    )

# 6. Monitoreo (opcional) -------------------------------------------------------------
if mostrar_monitoreo:
    st.subheader("Monitoreo de los datos de entrada (drift)")
    st.write("Compara los predictores publicados después del entrenamiento (desde agosto de "
             "2025) con los de febrero 2019 – julio 2025, con la prueba de Kolmogorov-Smirnov. "
             "Usa solo predictores, no ventas. Los meses seguidos no son independientes, así que "
             "los p-valores son orientativos.")
    tabla_drift = drift_predictores(df)
    st.markdown(f"**Semáforo:** {semaforo_drift(tabla_drift)}")
    st.dataframe(tabla_drift.round(3), hide_index=True, use_container_width=True)

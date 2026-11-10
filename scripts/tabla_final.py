# Dueño: Juanes
"""Tabla final con incertidumbre por bloques y figura de la prueba (entrega final, tareas 6 y 7).

Se corre DESPUÉS de la evaluación única en prueba, y no la vuelve a abrir: lee los CSV que
dejó scripts/evaluar_prueba.py (no llama a cargar_prueba).

    python scripts/tabla_final.py

Guarda:

    results/tablas/tabla_final.csv  una fila por modelo, escenario central:
        mape_cv_media, mape_cv_desv        MAPE de validación (5 folds) ± desviación entre folds
        mape_cv_min/max_escenarios         rango del MAPE de validación entre bajo, central y alto
        mape_prueba, ic_inf, ic_sup        MAPE nacional de prueba con IC 95 % por bloques de 3
                                           meses (bootstrap_bloques, B = 2000, semilla 42)
        mape_prueba_min/max_escenarios     rango del MAPE de prueba entre los tres escenarios
        dif_vs_ingenuo, ic_inf, ic_sup     MAPE(modelo) − MAPE(ingenuo estacional), pareada;
                                           negativa = el modelo se equivoca menos que el ingenuo.
                                           Es el criterio de la llamada de las 21:00 (piloto solo
                                           si el modelo supera al ingenuo con un IC que no cruza 0)
        dif_ridge_menos_knn, ic_inf, ic_sup  MAPE(Ridge) − MAPE(KNN), pareada (filas ridge y knn)
    results/figuras/fig_prueba_serie_nacional.png  serie nacional observada (con los dos años
        previos de entrenamiento como contexto) frente a Ridge, KNN y el ingenuo estacional.

Con 12 meses de prueba los intervalos salen anchos, y eso se declara en el informe.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    ESCENARIO_PRINCIPAL,
    INICIO_PRUEBA,
    LINEA_OBJETIVO_CODIGO,
    OBJETIVO,
    RUTA_TABLAS,
)
from src.datos.cargar import cargar_entrenamiento  # noqa: E402
from src.evaluacion.agregacion import agregar_nacional  # noqa: E402
from src.evaluacion.comparacion import INGENUO  # noqa: E402
from src.evaluacion.incertidumbre import (  # noqa: E402
    B_REPLICAS,
    LARGO_BLOQUE,
    bootstrap_bloques,
    bootstrap_diferencia,
)
from src.figuras import aplicar_estilo, guardar_figura  # noqa: E402

MODELOS = ["ridge", "knn", "mlr_stepwise", "media", INGENUO]
NOMBRES = {"ridge": "Ridge", "knn": "KNN (k = 3)", "mlr_stepwise": "MLR stepwise",
           "media": "Media", INGENUO: "Ingenuo estacional"}
# Paleta categórica de referencia (dataviz), validada: azul, naranja, aqua
COLORES = {"ridge": "#2a78d6", "knn": "#eb6834", INGENUO: "#1baf7a"}
MARCAS = {"ridge": "o", "knn": "s", INGENUO: "^"}
RUTA_SALIDA = RUTA_TABLAS


def _series(pred: pd.DataFrame, modelo: str) -> tuple[pd.Series, pd.Series]:
    """(observado, predicho) nacionales de la línea 8 en el escenario central, por fecha."""
    filas = pred[(pred["escenario"] == ESCENARIO_PRINCIPAL) & (pred["ambito"] == "Nacional")
                 & (pred["linea_codigo"] == LINEA_OBJETIVO_CODIGO) & (pred["modelo"] == modelo)]
    filas = filas.set_index(pd.to_datetime(filas["fecha"])).sort_index()
    return filas["y_real"], filas["y_pred"]


def construir_tabla_final(prueba: pd.DataFrame, predicciones: pd.DataFrame,
                          comparacion_cv: pd.DataFrame) -> pd.DataFrame:
    cv = comparacion_cv.set_index(["escenario", "modelo"])
    pr = prueba.set_index(["escenario", "modelo"])
    obs, pred_ingenuo = _series(predicciones, INGENUO)
    _, pred_ridge = _series(predicciones, "ridge")
    _, pred_knn = _series(predicciones, "knn")
    ridge_knn = bootstrap_diferencia(obs, pred_ridge, pred_knn)

    filas = []
    for modelo in MODELOS:
        obs_m, pred_m = _series(predicciones, modelo)
        if not obs_m.equals(obs):
            raise ValueError(f"{modelo}: el observado nacional no coincide con el del ingenuo")
        ic = bootstrap_bloques(obs_m, pred_m)
        vs_ingenuo = bootstrap_diferencia(obs_m, pred_m, pred_ingenuo)
        if abs(ic["mape"] - pr.loc[(ESCENARIO_PRINCIPAL, modelo), "mape"]) > 1e-4:
            raise ValueError(f"{modelo}: el MAPE recalculado no coincide con prueba.csv")
        mape_cv = cv.xs(modelo, level="modelo")["mape_media"]
        mape_pr = pr.xs(modelo, level="modelo")["mape"]
        fila = {
            "modelo": modelo,
            "mape_cv_media": cv.loc[(ESCENARIO_PRINCIPAL, modelo), "mape_media"],
            "mape_cv_desv": cv.loc[(ESCENARIO_PRINCIPAL, modelo), "mape_desv"],
            "mape_cv_min_escenarios": mape_cv.min(),
            "mape_cv_max_escenarios": mape_cv.max(),
            "mape_prueba": ic["mape"],
            "mape_prueba_ic_inf": ic["ic_inf"],
            "mape_prueba_ic_sup": ic["ic_sup"],
            "mape_prueba_min_escenarios": mape_pr.min(),
            "mape_prueba_max_escenarios": mape_pr.max(),
            "dif_vs_ingenuo": vs_ingenuo["dif"],
            "dif_vs_ingenuo_ic_inf": vs_ingenuo["ic_inf"],
            "dif_vs_ingenuo_ic_sup": vs_ingenuo["ic_sup"],
            "dif_ridge_menos_knn": None, "dif_ridge_menos_knn_ic_inf": None,
            "dif_ridge_menos_knn_ic_sup": None,
            "hiperparametros": pr.loc[(ESCENARIO_PRINCIPAL, modelo), "hiperparametros"],
            "n_meses": ic["n_meses"], "largo_bloque": LARGO_BLOQUE, "B": B_REPLICAS,
        }
        if modelo in ("ridge", "knn"):
            fila["dif_ridge_menos_knn"] = ridge_knn["dif"]
            fila["dif_ridge_menos_knn_ic_inf"] = ridge_knn["ic_inf"]
            fila["dif_ridge_menos_knn_ic_sup"] = ridge_knn["ic_sup"]
        filas.append(fila)
    return pd.DataFrame(filas)


def figura_prueba(predicciones: pd.DataFrame) -> plt.Figure:
    """Serie nacional observada frente a Ridge, KNN y el ingenuo, con 2 años previos de contexto."""
    aplicar_estilo()
    base = cargar_entrenamiento(ESCENARIO_PRINCIPAL)
    historia = agregar_nacional(base, OBJETIVO, LINEA_OBJETIVO_CODIGO)
    historia = historia[historia.index >= pd.Timestamp(INICIO_PRUEBA) - pd.DateOffset(months=24)]
    obs, _ = _series(predicciones, INGENUO)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(pd.Timestamp(INICIO_PRUEBA) - pd.Timedelta(days=15),
               obs.index.max() + pd.Timedelta(days=15), color="#f0efec", zorder=0)
    ax.text(pd.Timestamp(INICIO_PRUEBA), 0.015, "  prueba (ago 2025 – jul 2026)",
            transform=ax.get_xaxis_transform(), va="bottom", fontsize=9, color="#52514e")
    # Una sola línea observada: entrenamiento (contexto) y prueba seguidos, sin corte
    observado = pd.concat([historia, obs]).sort_index()
    ax.plot(observado.index, observado.to_numpy(), color="#0b0b0b", lw=2.2, label="Observado")
    for modelo in ("knn", "ridge", INGENUO):
        _, pred = _series(predicciones, modelo)
        ax.plot(pred.index, pred.to_numpy(), color=COLORES[modelo], lw=2,
                marker=MARCAS[modelo], ms=6, mec="#fcfcfb", mew=1.5, label=NOMBRES[modelo])
        ax.annotate(NOMBRES[modelo], (pred.index[-1], pred.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
    ax.annotate("Observado", (obs.index[-1], obs.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
    ax.set_ylabel("Índice de ventas reales, línea 8 (2019 = 100)")
    ax.set_title("Serie nacional en la prueba: los tres modelos se quedan cortos todos los meses",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax.set_xlim(historia.index.min(), obs.index.max() + pd.DateOffset(months=4))
    ax.set_ylim(bottom=92)  # deja aire abajo para la etiqueta de la prueba
    fig.tight_layout()
    return fig


def main() -> pd.DataFrame:
    ruta = Path(RUTA_SALIDA)
    prueba = pd.read_csv(ruta / "prueba.csv")
    predicciones = pd.read_csv(ruta / "prueba_predicciones.csv")
    comparacion_cv = pd.read_csv(ruta / "comparacion_cv.csv")
    tabla = construir_tabla_final(prueba, predicciones, comparacion_cv)
    tabla.to_csv(ruta / "tabla_final.csv", index=False, float_format="%.6f")
    guardar_figura(figura_prueba(predicciones), "fig_prueba_serie_nacional")
    columnas = ["modelo", "mape_cv_media", "mape_prueba", "mape_prueba_ic_inf",
                "mape_prueba_ic_sup", "dif_vs_ingenuo", "dif_vs_ingenuo_ic_inf",
                "dif_vs_ingenuo_ic_sup"]
    print(tabla[columnas].round(2).to_string(index=False))
    fila = tabla.set_index("modelo").loc["ridge"]
    inf, sup = fila["dif_ridge_menos_knn_ic_inf"], fila["dif_ridge_menos_knn_ic_sup"]
    print(f"Ridge − KNN: {fila['dif_ridge_menos_knn']:.2f} (IC {inf:.2f} a {sup:.2f})")
    return tabla


if __name__ == "__main__":
    main()

# Dueño: Juanes
"""Evaluación ÚNICA en la prueba (agosto 2025 – julio 2026). Es el único archivo que lee la
prueba (cargar_prueba); lo vigila tests/test_fuga.py.

Entrega final. Se deja listo el lunes 9 (tarea 4) y se corre UNA sola vez el martes 10
entre las 12:00 y las 14:00, solo Juanes, y solo después de que el commit de congelamiento
(docs/especificacion_final.md) esté en main:

    python scripts/evaluar_prueba.py --escenario todos          # LA corrida oficial
    python scripts/evaluar_prueba.py --escenario todos --rezagos rez1   # solo si se pide

Si results/tablas/prueba.csv ya existe, el script se niega a correr: la prueba se abre una
vez. Para repetir (por ejemplo, el clon limpio de Aleja que compara hashes) se pasa --forzar.

Qué hace, para cada escenario:

1. Carga entrenamiento (feb 2019 – jul 2025) y prueba (ago 2025 – jul 2026), los CONCATENA
   y aplica preparar_datos (agregar_rezagos_reales de Julián), y luego los vuelve a separar
   por INICIO_PRUEBA. Por qué concatenar: el rezago de los primeros meses de prueba está en
   los últimos de entrenamiento (ejemplo: importaciones de agosto 2025 con k = 3 es el dato
   de mayo 2025). El desplazamiento solo mira al pasado de cada serie departamento × línea,
   así que ningún dato de la prueba entra a una fila de entrenamiento.
2. Entrena con TODO el entrenamiento (menos los meses que descarta el rezago):
   - Ridge, KNN y MLR con stepwise: buscar_hiperparametros (validar.py), la MISMA búsqueda de
     entrenar.py (GridSearchCV + CVTemporalPorMes + scorer_mape_nacional, rejilla congelada)
     y refit=True. Los hiperparámetros los elige la validación cruzada, nunca la prueba.
   - Media e ingenuo estacional: fit directo.
3. Predice los 12 meses de prueba y mide, en la serie nacional de la línea 8 (agregada con
   agregar_nacional), MAPE, MAD, MSD y RMSE; y el MAPE del panel (las 56 series).
4. Guarda (results/tablas/; con --rezagos rez1 llevan el sufijo _rez1):
   prueba.csv               escenario, modelo, métricas de prueba, mejora % frente a la media
                            y al ingenuo estacional, MAPE de validación (media y desviación
                            entre folds) e hiperparámetros
   prueba_predicciones.csv  escenario, fecha, ambito, linea_codigo, modelo, y_real, y_pred
                            (ambito = departamento, más las filas Nacional de la línea 8)
   prueba_pareada.csv       mes a mes, Ridge frente a KNN en la serie nacional: lo que
                            necesita la regla del modelo final y el bootstrap pareado
5. Imprime la regla Ridge/KNN (regla_modelo_final) en cada escenario. La que decide es la
   del central (ESCENARIO_PRINCIPAL); su resultado se escribe en docs/decisiones.md.

Un solo hilo (threadpool_limits), como entrenar.py, para que KNN dé igual en cualquier PC.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    ESCENARIO_PRINCIPAL,
    ESCENARIOS,
    INICIO_PRUEBA,
    LINEA_OBJETIVO_CODIGO,
    OBJETIVO,
    RUTA_TABLAS,
)
from src.datos.cargar import cargar_entrenamiento, cargar_prueba  # noqa: E402
from src.evaluacion.agregacion import agregar_nacional  # noqa: E402
from src.evaluacion.comparacion import INGENUO, MEDIA, regla_modelo_final  # noqa: E402
from src.evaluacion.metricas import mad, mape, mejora_pct, msd, rmse  # noqa: E402
from src.evaluacion.particion import CVTemporalPorMes  # noqa: E402
from src.evaluacion.validar import (  # noqa: E402
    buscar_hiperparametros,
    evaluar_cv,
    mape_cv_ganador,
)
from src.modelos.candidatos import prefijo  # noqa: E402
from src.modelos.lineas_base import IngenuoEstacional, PredictorMedia  # noqa: E402
from src.modelos.pipeline import REZAGOS, preparar_datos  # noqa: E402

# Los modelos que se evalúan en la prueba (sección 6, tarea 4): los dos candidatos a modelo
# final, el método del artículo y las dos líneas base. Lasso y la MLR completa quedan solo
# en la validación cruzada: la regla congelada no los considera.
CANDIDATOS_PRUEBA = ["ridge", "knn", "mlr_stepwise"]
LINEAS_BASE = {MEDIA: PredictorMedia, INGENUO: IngenuoEstacional}
NACIONAL = "Nacional"
N_MESES_PRUEBA = 12

COLUMNAS_PRUEBA = ["escenario", "modelo", "mape", "mad", "msd", "rmse", "mape_panel",
                   "mejora_vs_media", "mejora_vs_ingenuo", "mape_cv_media", "mape_cv_desv",
                   "hiperparametros", "n_meses"]
COLUMNAS_PREDICCIONES = ["escenario", "fecha", "ambito", "linea_codigo", "modelo",
                         "y_real", "y_pred"]
COLUMNAS_PAREADA = ["escenario", "fecha", "y_real", "pred_ridge", "pred_knn",
                    "error_abs_ridge", "error_abs_knn", "knn_gana"]

# Se puede cambiar desde las pruebas (monkeypatch) para no escribir en results/tablas
RUTA_SALIDA = RUTA_TABLAS


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def separar_con_rezagos(entrena: pd.DataFrame, prueba: pd.DataFrame,
                        rezagos: str = "reales") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(entrenamiento, prueba, predictores) con los rezagos aplicados sobre la serie completa.

    Concatena, aplica preparar_datos (rezagos de Julián + orden fijo de filas) y vuelve a
    separar por INICIO_PRUEBA. Comprueba que la prueba tenga sus 12 meses completos (ningún
    mes de prueba se pierde por el rezago) y que las dos partes no se crucen en el tiempo.
    """
    inicio = pd.Timestamp(INICIO_PRUEBA)
    if (pd.to_datetime(entrena["fecha"]) >= inicio).any():
        raise ValueError("el entrenamiento trae meses de la prueba")
    if (pd.to_datetime(prueba["fecha"]) < inicio).any():
        raise ValueError("la prueba trae meses anteriores a INICIO_PRUEBA")
    junto = pd.concat([entrena, prueba], ignore_index=True)
    datos, predictores = preparar_datos(junto, rezagos)
    es_prueba = pd.to_datetime(datos["fecha"]) >= inicio
    ent = datos[~es_prueba].reset_index(drop=True)
    pru = datos[es_prueba].reset_index(drop=True)
    if len(pru) != len(prueba):
        raise ValueError(f"la prueba tenía {len(prueba)} filas y quedaron {len(pru)} "
                         "después de los rezagos: a algún mes le falta su dato rezagado")
    return ent, pru, predictores


# ---------------------------------------------------------------------------
# Ajuste y predicción
# ---------------------------------------------------------------------------
def _texto(params: dict) -> str:
    if not params:
        return "sin hiperparámetros"
    partes = []
    for nombre, valor in sorted(params.items()):
        if isinstance(valor, float | np.floating):
            valor = f"{valor:.4g}"
        partes.append(f"{nombre.removeprefix(prefijo())}={valor}")
    return "; ".join(partes)


def ajustar_modelos(ent: pd.DataFrame, predictores: list[str]) -> dict:
    """{nombre: (modelo ajustado con todo el entrenamiento, MAPE de CV por fold, texto)}."""
    cv = CVTemporalPorMes()
    modelos = {}
    for nombre, clase in LINEAS_BASE.items():
        # Su MAPE de validación con los mismos folds, para la columna mape_cv de prueba.csv
        mape_folds = evaluar_cv(clase(), ent, cv)["mape"].to_numpy()
        modelos[nombre] = (clase().fit(ent, ent[OBJETIVO]), mape_folds, _texto({}))
    for nombre in CANDIDATOS_PRUEBA:
        busqueda = buscar_hiperparametros(nombre, ent, predictores, cv=cv)
        modelos[nombre] = (busqueda.best_estimator_, mape_cv_ganador(busqueda),
                           _texto(busqueda.best_params_))
    return modelos


def _nacional(pru: pd.DataFrame, y_pred) -> tuple[pd.Series, pd.Series]:
    datos = pru[["fecha", "departamento", "linea_codigo"]].copy()
    datos["y_real"] = pru[OBJETIVO].to_numpy(dtype=float)
    datos["y_pred"] = np.asarray(y_pred, dtype=float)
    return (agregar_nacional(datos, "y_real", LINEA_OBJETIVO_CODIGO),
            agregar_nacional(datos, "y_pred", LINEA_OBJETIVO_CODIGO))


def evaluar_escenario(escenario: str, ent: pd.DataFrame, pru: pd.DataFrame,
                      predictores: list[str]) -> dict[str, pd.DataFrame]:
    """Las tres tablas de un escenario. No lee archivos: recibe los datos ya separados."""
    modelos = ajustar_modelos(ent, predictores)
    metricas, predicciones, nacionales = [], [], {}
    y_panel = pru[OBJETIVO].to_numpy(dtype=float)
    for nombre, (modelo, mape_folds, texto) in modelos.items():
        pred = modelo.predict(pru)
        obs_nac, pred_nac = _nacional(pru, pred)
        nacionales[nombre] = (obs_nac, pred_nac)
        metricas.append({
            "escenario": escenario, "modelo": nombre,
            "mape": mape(obs_nac, pred_nac), "mad": mad(obs_nac, pred_nac),
            "msd": msd(obs_nac, pred_nac), "rmse": rmse(obs_nac, pred_nac),
            "mape_panel": mape(y_panel, pred),
            "mape_cv_media": float(np.mean(mape_folds)),
            "mape_cv_desv": float(np.std(mape_folds, ddof=1)),
            "hiperparametros": texto, "n_meses": int(obs_nac.size),
        })
        panel = pd.DataFrame({
            "escenario": escenario, "fecha": pru["fecha"].to_numpy(),
            "ambito": pru["departamento"].to_numpy(),
            "linea_codigo": pru["linea_codigo"].to_numpy(), "modelo": nombre,
            "y_real": y_panel, "y_pred": pred,
        })
        nacional = pd.DataFrame({
            "escenario": escenario, "fecha": obs_nac.index, "ambito": NACIONAL,
            "linea_codigo": LINEA_OBJETIVO_CODIGO, "modelo": nombre,
            "y_real": obs_nac.to_numpy(), "y_pred": pred_nac.to_numpy(),
        })
        predicciones += [panel, nacional]

    tabla = pd.DataFrame(metricas)
    ref = tabla.set_index("modelo")["mape"]
    tabla["mejora_vs_media"] = [mejora_pct(m, ref[MEDIA]) for m in tabla["mape"]]
    tabla["mejora_vs_ingenuo"] = [mejora_pct(m, ref[INGENUO]) for m in tabla["mape"]]

    obs, ridge = nacionales["ridge"]
    knn = nacionales["knn"][1]
    pareada = pd.DataFrame({
        "escenario": escenario, "fecha": obs.index, "y_real": obs.to_numpy(),
        "pred_ridge": ridge.to_numpy(), "pred_knn": knn.to_numpy(),
    })
    pareada["error_abs_ridge"] = (pareada["y_real"] - pareada["pred_ridge"]).abs()
    pareada["error_abs_knn"] = (pareada["y_real"] - pareada["pred_knn"]).abs()
    pareada["knn_gana"] = pareada["error_abs_knn"] < pareada["error_abs_ridge"]
    return {
        "prueba": tabla[COLUMNAS_PRUEBA],
        "prueba_predicciones": pd.concat(predicciones, ignore_index=True)[COLUMNAS_PREDICCIONES],
        "prueba_pareada": pareada[COLUMNAS_PAREADA],
    }


def regla_desde_tablas(prueba: pd.DataFrame, pareada: pd.DataFrame, escenario: str) -> dict:
    """regla_modelo_final con las cifras de prueba.csv y prueba_pareada.csv de un escenario."""
    par = pareada[pareada["escenario"] == escenario]
    desv = prueba.set_index(["escenario", "modelo"]).loc[(escenario, "ridge"), "mape_cv_desv"]
    return regla_modelo_final(par["y_real"], par["pred_ridge"], par["pred_knn"], desv)


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escenario", default=ESCENARIO_PRINCIPAL,
                        choices=[*ESCENARIOS, "todos"])
    parser.add_argument("--rezagos", default="reales", choices=list(REZAGOS),
                        help="reales (por defecto, especificación final) o rez1 (Entrega 2)")
    parser.add_argument("--forzar", action="store_true",
                        help="volver a correr aunque prueba.csv ya exista (solo el clon limpio)")
    args = parser.parse_args(argv)

    sufijo = "" if args.rezagos == "reales" else f"_{args.rezagos}"
    ruta = Path(RUTA_SALIDA)
    if (ruta / f"prueba{sufijo}.csv").exists() and not args.forzar:
        raise SystemExit(f"{ruta / f'prueba{sufijo}.csv'} ya existe: la prueba se evalúa una "
                         "sola vez. Usa --forzar solo para reproducir (clon limpio).")

    escenarios = list(ESCENARIOS) if args.escenario == "todos" else [args.escenario]
    inicio = time.perf_counter()
    por_tabla: dict[str, list] = {"prueba": [], "prueba_predicciones": [], "prueba_pareada": []}
    with threadpool_limits(limits=1):
        for escenario in escenarios:
            print(f"Escenario {escenario}, rezagos {args.rezagos}")
            ent, pru, predictores = separar_con_rezagos(
                cargar_entrenamiento(escenario), cargar_prueba(escenario), args.rezagos)
            meses = pd.to_datetime(pru["fecha"]).nunique()
            if meses != N_MESES_PRUEBA:
                print(f"  Aviso: la prueba tiene {meses} meses, no {N_MESES_PRUEBA}")
            for nombre, tabla in evaluar_escenario(escenario, ent, pru, predictores).items():
                por_tabla[nombre].append(tabla)

    ruta.mkdir(parents=True, exist_ok=True)
    opciones = {"index": False, "float_format": "%.6f", "date_format": "%Y-%m-%d"}
    tablas = {n: pd.concat(t, ignore_index=True) for n, t in por_tabla.items()}
    for nombre, tabla in tablas.items():
        tabla.to_csv(ruta / f"{nombre}{sufijo}.csv", **opciones)

    for escenario in escenarios:
        r = regla_desde_tablas(tablas["prueba"], tablas["prueba_pareada"], escenario)
        marca = "  ← decide" if escenario == ESCENARIO_PRINCIPAL else ""
        print(f"  Regla [{escenario}]: Ridge {r['mape_ridge']:.2f} %, KNN {r['mape_knn']:.2f} %"
              f", desv. CV Ridge {r['desv_cv_ridge']:.2f}; (a) {r['condicion_a']}, "
              f"(b) KNN gana {r['meses_knn_gana']}/{r['n_meses']} → {r['modelo_final']}{marca}")
    print(f"Listo en {time.perf_counter() - inicio:.0f} s. Tablas en {ruta}")
    return tablas


if __name__ == "__main__":
    main()

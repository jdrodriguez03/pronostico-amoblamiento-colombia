# Dueño: Juanes
"""Tablas del modelo final que salen SOLO de la validación cruzada (entrega final, tarea 5).

No lee la prueba. Se corre después de entrenar.py con las dos especificaciones:

    python scripts/entrenar.py --escenario todos --rezagos rez1
    python scripts/entrenar.py --escenario todos --rezagos reales
    python scripts/preparar_modelo_final.py

Guarda en results/tablas/:

    comparacion_cv_rezagos.csv  escenario central: MAPE de validación de Ridge, KNN, MLR con
                                stepwise y las dos líneas base con rez1 y con rezagos reales,
                                y la diferencia («cuánto cuesta ser realistas»)
    predicciones_oof.csv        predicciones fuera de muestra de los 5 folds (30 meses) de
                                Ridge y KNN con ESPECIFICACION_FINAL, escenario central
    rango_error.csv             percentiles 10, 50 y 90 del error relativo fuera de muestra
                                del modelo final, por ámbito y línea (lo usa la app)

Antes de escribir comprueba dos cosas, para que nada quede desalineado:
- que los hiperparámetros de ESPECIFICACION_FINAL sean los que eligió la validación
  cruzada en comparacion_cv.csv (escenario central);
- que el MAPE nacional de cada fold recalculado con las predicciones fuera de muestra
  coincida con cv_resultados.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ESCENARIO_PRINCIPAL, LINEA_OBJETIVO_CODIGO, RUTA_TABLAS  # noqa: E402
from src.datos.cargar import cargar_entrenamiento  # noqa: E402
from src.evaluacion.agregacion import agregar_nacional  # noqa: E402
from src.evaluacion.comparacion import tabla_costo_rezagos  # noqa: E402
from src.evaluacion.incertidumbre import rango_error  # noqa: E402
from src.evaluacion.metricas import mape  # noqa: E402
from src.evaluacion.particion import CVTemporalPorMes  # noqa: E402
from src.evaluacion.validar import predicciones_oof  # noqa: E402
from src.modelos.final import (  # noqa: E402
    ESPECIFICACION_FINAL,
    HIPERPARAMETROS,
    MODELO_FINAL,
    construir_modelo_final,
)
from src.modelos.pipeline import preparar_datos  # noqa: E402

RUTA_SALIDA = RUTA_TABLAS


def _texto(nombre: str) -> str:
    """Los hiperparámetros como los escribe entrenar.py: 'alpha=316.2', 'n_neighbors=3; ...'."""
    partes = []
    for clave, valor in sorted(HIPERPARAMETROS[nombre].items()):
        partes.append(f"{clave}={valor:.4g}" if isinstance(valor, float) else f"{clave}={valor}")
    return "; ".join(partes)


def comprobar_hiperparametros(comparacion: pd.DataFrame) -> None:
    """Falla si ESPECIFICACION_FINAL no es lo que eligió la validación cruzada."""
    central = comparacion[comparacion["escenario"] == ESCENARIO_PRINCIPAL].set_index("modelo")
    for nombre in HIPERPARAMETROS:
        elegido = central.loc[nombre, "hiperparametros"]
        if elegido != _texto(nombre):
            raise ValueError(f"{nombre}: la validación eligió {elegido!r} pero final.py dice "
                             f"{_texto(nombre)!r}. Actualiza final.py o vuelve a entrenar.")


def mape_por_fold(oof: pd.DataFrame) -> pd.Series:
    """MAPE nacional (línea 8) de cada fold con las predicciones fuera de muestra."""
    salida = {}
    for fold, grupo in oof.groupby("fold"):
        obs = agregar_nacional(grupo, "y_real", LINEA_OBJETIVO_CODIGO)
        pred = agregar_nacional(grupo, "y_pred", LINEA_OBJETIVO_CODIGO)
        salida[fold] = mape(obs, pred)
    return pd.Series(salida, name="mape")


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modelo", default=MODELO_FINAL, choices=sorted(HIPERPARAMETROS),
                        help="modelo final para rango_error.csv (lo decide la regla)")
    args = parser.parse_args(argv)
    ruta = Path(RUTA_SALIDA)

    comparacion_reales = pd.read_csv(ruta / "comparacion_cv.csv")
    comparacion_rez1 = pd.read_csv(ruta / "comparacion_cv_rez1.csv")
    cv_reales = pd.read_csv(ruta / "cv_resultados.csv")
    comprobar_hiperparametros(comparacion_reales)
    costo = tabla_costo_rezagos(comparacion_rez1, comparacion_reales, ESCENARIO_PRINCIPAL)

    escenario = ESPECIFICACION_FINAL["escenario"]
    datos, _ = preparar_datos(cargar_entrenamiento(escenario),
                              ESPECIFICACION_FINAL["rezagos_nombre"])
    cv = CVTemporalPorMes()
    partes = []
    with threadpool_limits(limits=1):
        for nombre in HIPERPARAMETROS:
            oof = predicciones_oof(construir_modelo_final(nombre), datos, cv)
            # Comprobación cruzada con entrenar.py: mismos folds, mismo MAPE
            esperado = (cv_reales[(cv_reales["escenario"] == escenario)
                                  & (cv_reales["modelo"] == nombre)]
                        .set_index("fold")["mape"].sort_index())
            obtenido = mape_por_fold(oof)
            # El CSV guarda 6 decimales: se compara con esa precisión
            if not np.allclose(obtenido.to_numpy(), esperado.to_numpy(), rtol=0, atol=1e-5):
                raise ValueError(f"{nombre}: MAPE por fold {obtenido.round(4).tolist()} ≠ "
                                 f"cv_resultados.csv {esperado.round(4).tolist()}")
            print(f"{nombre:6s} MAPE por fold {obtenido.round(2).tolist()} (= cv_resultados)")
            partes.append(oof.assign(escenario=escenario, modelo=nombre))
    oof_todos = pd.concat(partes, ignore_index=True)
    oof_todos = oof_todos[["escenario", "modelo", "fold", "fecha", "departamento",
                           "linea_codigo", "y_real", "y_pred"]]
    rango = rango_error(oof_todos[oof_todos["modelo"] == args.modelo].drop(columns="modelo"))

    opciones = {"index": False, "float_format": "%.6f", "date_format": "%Y-%m-%d"}
    costo.to_csv(ruta / "comparacion_cv_rezagos.csv", **opciones)
    oof_todos.to_csv(ruta / "predicciones_oof.csv", **opciones)
    rango.to_csv(ruta / "rango_error.csv", **opciones)
    print(costo[["modelo", "mape_media_rez1", "mape_media_reales", "diferencia"]]
          .round(2).to_string(index=False))
    nacional = rango[(rango["ambito"] == "Nacional")
                     & (rango["linea_codigo"] == LINEA_OBJETIVO_CODIGO)]
    print(f"rango_error.csv con {args.modelo}: {len(rango)} filas; Nacional línea 8:\n"
          f"{nacional.round(4).to_string(index=False)}")
    return {"comparacion_cv_rezagos": costo, "predicciones_oof": oof_todos, "rango_error": rango}


if __name__ == "__main__":
    main()

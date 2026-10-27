# Dueño: Juanes
"""Validación cruzada de todos los candidatos y las líneas base.

Plan de cierre, lunes 26, tareas 2 y 3 (camino crítico).

Uso:
    python scripts/entrenar.py --escenario todos      # corrida oficial (3 escenarios)
    python scripts/entrenar.py --escenario central    # un solo escenario
    python scripts/entrenar.py --rapido               # base pequeña, para probar (< 1 min)

Qué hace, para cada escenario:

1. Carga SOLO el entrenamiento (febrero de 2019 a julio de 2025) con cargar_entrenamiento.
   La prueba (agosto de 2025 a julio de 2026) no se lee aquí: solo evaluar_prueba.py.
2. Para cada uno de los 5 candidatos de candidatos.py (Julián):
       GridSearchCV(pipeline, rejilla, cv=CVTemporalPorMes(), scoring=scorer_mape_nacional,
                    refit=True, return_train_score=True)
   - cv=CVTemporalPorMes(): 5 folds de ventana creciente por MES (regla de oro 1).
   - scoring=scorer_mape_nacional: elige los hiperparámetros por el MAPE de la serie
     nacional de muebles, la misma cifra que reporta el artículo.
   - refit=True: con los hiperparámetros ganadores reentrena con los 78 meses
     (best_estimator_); de ahí sale mape_ajuste.
   - return_train_score=True: guarda también el MAPE de entrenamiento de cada fold.
   Todo (escalado, codificación, stepwise, penalización) va dentro del Pipeline, así que en
   cada fold se aprende solo con los meses de entrenamiento de ese fold (regla de oro 2).
3. Con los hiperparámetros ganadores pasa el modelo por evaluar_cv para obtener, fold por
   fold, MAPE, MAD, MSD, RMSE nacionales, MAPE del panel y MAPE de entrenamiento. Comprueba
   que el MAPE de validación y el de entrenamiento coinciden con los que guardó
   GridSearchCV: así las líneas base y los modelos quedan medidos exactamente igual.
4. Las dos líneas base (media e ingenuo estacional) no tienen hiperparámetros: entran
   directo por evaluar_cv.

Salidas (results/tablas/):
    cv_resultados.csv        una fila por escenario, modelo y fold (7 × 3 × 5 = 105)
    comparacion_cv.csv       una fila por escenario y modelo (21)
    comparacion_pareada.csv  diferencias de MAPE fold por fold frente a las referencias
    rango_escenarios.csv     MAPE medio de cada modelo en los 3 escenarios, rango y puesto
y las búsquedas ajustadas en results/modelos/ (en .gitignore), para que Julián dibuje las
curvas de validación sin volver a entrenar.

Reproducibilidad: todo corre en UN solo hilo y sin procesos en paralelo. Motivo (26 de
octubre): en KNN hay empates exactos de distancia (las 56 filas de un mes comparten los
predictores nacionales, así que dos vecinos de líneas distintas del mismo mes quedan a la
misma distancia). Cuál de los empatados entra como k-ésimo vecino lo decide el redondeo, y
el redondeo cambia con el número de hilos del PC: con 2 hilos el MAPE del fold 3 de KNN
(escenario bajo) salía 6,75 % y con 1 hilo 5,51 %. Con un solo hilo el resultado es el
mismo en cualquier computador. La corrida completa tarda menos de un minuto, así que el
paralelismo no hace falta.

Con --rapido usa la base pequeña de data/fixtures y rejillas de 2 valores, y escribe todo
en results/modelos/rapido/ para no pisar las tablas oficiales.
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

# Para que "python scripts/entrenar.py" encuentre el paquete src aunque no se haya hecho
# pip install -e . (el CI sí lo hace; esto es solo comodidad al correrlo a mano)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ESCENARIOS, OBJETIVO, RUTA_MODELOS, RUTA_TABLAS  # noqa: E402
from src.datos.cargar import cargar_entrenamiento  # noqa: E402
from src.evaluacion.comparacion import (  # noqa: E402
    INGENUO,
    MEDIA,
    tabla_comparacion,
    tabla_pareada,
    tabla_rango_escenarios,
)
from src.evaluacion.particion import CVTemporalPorMes  # noqa: E402
from src.evaluacion.validar import (  # noqa: E402
    evaluar_cv,
    mape_ajuste,
    scorer_mape_nacional,
)
from src.modelos.candidatos import CANDIDATOS, construir_candidato, prefijo  # noqa: E402
from src.modelos.lineas_base import IngenuoEstacional, PredictorMedia  # noqa: E402

# Orden de las columnas de cv_resultados.csv
COLUMNAS = ["escenario", "modelo", "fold", "valida_desde", "valida_hasta", "hiperparametros",
            "mape", "mad", "msd", "rmse", "mape_panel", "mape_entrena", "mape_ajuste"]

# Tolerancia de la comprobación GridSearchCV == evaluar_cv. Los dos hacen exactamente las
# mismas cuentas; solo pueden diferir por redondeo de punto flotante.
TOLERANCIA = 1e-8


# ---------------------------------------------------------------------------
# Ayudantes
# ---------------------------------------------------------------------------
def rejilla_rapida(rejilla: dict) -> dict:
    """Deja 2 valores por hiperparámetro (el primero y el último) para el modo --rapido.

    Se toman los extremos para que la búsqueda siga teniendo algo que elegir y se pruebe
    el camino completo del código, pero en una fracción del tiempo.
    """
    return {k: [v[0], v[-1]] if len(v) > 1 else list(v) for k, v in rejilla.items()}


def texto_hiperparametros(params: dict) -> str:
    """{"regressor__modelo__alpha": 0.316} → "alpha=0.316" (legible en la tabla)."""
    if not params:
        return "sin hiperparámetros"
    partes = []
    for nombre, valor in sorted(params.items()):
        corto = nombre.removeprefix(prefijo())
        if isinstance(valor, float | np.floating):
            valor = f"{valor:.4g}"
        partes.append(f"{corto}={valor}")
    return "; ".join(partes)


def _comprobar_igual_a_gridsearch(busqueda: GridSearchCV, por_fold: pd.DataFrame,
                                  nombre: str) -> None:
    """MAPE de validación y de entrenamiento de evaluar_cv == los de GridSearchCV.

    GridSearchCV guarda el score de cada fold como −MAPE (scikit-learn maximiza), por eso
    se voltea el signo. Si no coincidieran, cv_resultados mezclaría dos formas de medir.
    """
    i = busqueda.best_index_
    res = busqueda.cv_results_
    n = busqueda.n_splits_
    valida = np.array([-res[f"split{k}_test_score"][i] for k in range(n)])
    entrena = np.array([-res[f"split{k}_train_score"][i] for k in range(n)])
    if not (np.allclose(valida, por_fold["mape"], rtol=TOLERANCIA, atol=TOLERANCIA)
            and np.allclose(entrena, por_fold["mape_entrena"], rtol=TOLERANCIA,
                            atol=TOLERANCIA)):
        raise RuntimeError(f"{nombre}: evaluar_cv no coincide con GridSearchCV")


# ---------------------------------------------------------------------------
# Entrenamiento de un escenario
# ---------------------------------------------------------------------------
def entrenar_escenario(escenario: str, rapido: bool = False,
                       ruta_modelos: Path = RUTA_MODELOS) -> pd.DataFrame:
    """cv_resultados de un escenario: 7 modelos × 5 folds."""
    df = cargar_entrenamiento(escenario, fixture=rapido)
    y = df[OBJETIVO]
    cv = CVTemporalPorMes()
    ruta_modelos.mkdir(parents=True, exist_ok=True)
    tablas = []

    # 1. Las líneas base: sin hiperparámetros, directo por evaluar_cv
    for nombre, modelo in [(MEDIA, PredictorMedia()), (INGENUO, IngenuoEstacional())]:
        inicio = time.perf_counter()
        por_fold = evaluar_cv(modelo, df, cv)
        ajustado = modelo.fit(df, y)  # con los 78 meses, igual que refit=True
        por_fold["hiperparametros"] = texto_hiperparametros({})
        por_fold["mape_ajuste"] = mape_ajuste(ajustado, df)
        tablas.append(por_fold.assign(escenario=escenario, modelo=nombre))
        joblib.dump(ajustado, ruta_modelos / f"{escenario}_{nombre}.joblib")
        print(f"  {nombre:20} MAPE {por_fold['mape'].mean():6.2f} %  "
              f"({time.perf_counter() - inicio:.0f} s)")

    # 2. Los 5 candidatos: GridSearchCV elige hiperparámetros por el MAPE nacional
    for nombre in CANDIDATOS:
        inicio = time.perf_counter()
        pipeline, rejilla = construir_candidato(nombre)
        if rapido:
            rejilla = rejilla_rapida(rejilla)
        # Sin rejilla (MLR stepwise y completa), {} es "probar solo la configuración por
        # defecto": GridSearchCV hace un único ajuste por fold y sirve igual
        busqueda = GridSearchCV(
            pipeline, rejilla, cv=cv, scoring=scorer_mape_nacional,
            refit=True, return_train_score=True, n_jobs=1,
        ).fit(df, y)

        # Métricas por fold con los hiperparámetros ganadores (mismos folds, mismas cuentas)
        elegido = construir_candidato(nombre)[0].set_params(**busqueda.best_params_)
        por_fold = evaluar_cv(elegido, df, cv)
        _comprobar_igual_a_gridsearch(busqueda, por_fold, nombre)

        por_fold["hiperparametros"] = texto_hiperparametros(busqueda.best_params_)
        por_fold["mape_ajuste"] = mape_ajuste(busqueda.best_estimator_, df)
        tablas.append(por_fold.assign(escenario=escenario, modelo=nombre))
        joblib.dump(busqueda, ruta_modelos / f"{escenario}_{nombre}.joblib")
        print(f"  {nombre:20} MAPE {por_fold['mape'].mean():6.2f} %  "
              f"[{por_fold['hiperparametros'].iloc[0]}]  "
              f"({time.perf_counter() - inicio:.0f} s)")

    return pd.concat(tablas, ignore_index=True)[COLUMNAS]


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def _guardar(cv_nuevo: pd.DataFrame, ruta_tablas: Path) -> pd.DataFrame:
    """Escribe las tres tablas. Si ya existe cv_resultados.csv, reemplaza solo los
    escenarios que se acaban de correr y conserva los demás."""
    ruta_tablas.mkdir(parents=True, exist_ok=True)
    ruta_cv = ruta_tablas / "cv_resultados.csv"
    if ruta_cv.exists():
        anterior = pd.read_csv(ruta_cv, parse_dates=["valida_desde", "valida_hasta"])
        anterior = anterior[~anterior["escenario"].isin(cv_nuevo["escenario"].unique())]
        cv_todo = pd.concat([anterior, cv_nuevo], ignore_index=True)
    else:
        cv_todo = cv_nuevo
    # Orden fijo: escenarios como en config, primero las líneas base y luego los candidatos
    orden = {"escenario": list(ESCENARIOS), "modelo": [MEDIA, INGENUO, *CANDIDATOS]}
    cv_todo = (cv_todo
               .sort_values(["escenario", "modelo", "fold"],
                            key=lambda c: c.map({v: i for i, v in enumerate(orden[c.name])})
                            if c.name in orden else c)
               .reset_index(drop=True))

    # Fechas como AAAA-MM-DD y 6 decimales: el CSV queda estable entre corridas y PCs
    opciones = {"index": False, "float_format": "%.6f", "date_format": "%Y-%m-%d"}
    cv_todo.to_csv(ruta_cv, **opciones)
    comparacion = tabla_comparacion(cv_todo)
    comparacion.to_csv(ruta_tablas / "comparacion_cv.csv", **opciones)
    # Martes 27: el rango entre escenarios (variabilidad de la 7.4) también sale de un CSV
    tabla_rango_escenarios(comparacion).to_csv(ruta_tablas / "rango_escenarios.csv",
                                               **opciones)
    tabla_pareada(cv_todo).to_csv(ruta_tablas / "comparacion_pareada.csv", **opciones)
    return cv_todo


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escenario", default="central",
                        choices=[*ESCENARIOS, "todos"],
                        help="escenario de η a entrenar (por defecto: central)")
    parser.add_argument("--rapido", action="store_true",
                        help="base pequeña y rejillas de 2 valores (para probar el código)")
    args = parser.parse_args(argv)

    escenarios = list(ESCENARIOS) if args.escenario == "todos" else [args.escenario]
    # En modo rápido todo va a una carpeta ignorada por git: nunca pisa las tablas oficiales
    ruta_modelos = RUTA_MODELOS / "rapido" if args.rapido else RUTA_MODELOS
    ruta_tablas = ruta_modelos if args.rapido else RUTA_TABLAS

    inicio = time.perf_counter()
    tablas = []
    # Un solo hilo en numpy/BLAS/OpenMP: ver "Reproducibilidad" arriba
    with threadpool_limits(limits=1):
        for escenario in escenarios:
            print(f"Escenario {escenario}{' (rápido)' if args.rapido else ''}")
            tablas.append(entrenar_escenario(escenario, args.rapido, ruta_modelos))
    cv_todo = _guardar(pd.concat(tablas, ignore_index=True), ruta_tablas)
    print(f"Listo en {time.perf_counter() - inicio:.0f} s. Tablas en {ruta_tablas}")
    return cv_todo


if __name__ == "__main__":
    main()

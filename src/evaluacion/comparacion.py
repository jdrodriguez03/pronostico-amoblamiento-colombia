# Dueño: Juanes
"""Tablas de comparación de modelos a partir de cv_resultados.csv.

Plan de cierre, lunes 26, tarea 3. Dos tablas, las dos se calculan SOLO con los folds de
validación cruzada (meses de entrenamiento, febrero de 2019 a julio de 2025); la prueba no
se abre en esta entrega.

1. tabla_comparacion(): una fila por escenario y modelo (3 × 7 = 21 filas). Resume los
   5 folds de cada modelo:
     - mape_media ± mape_desv: el MAPE de validación promedio y su desviación entre folds.
       La desviación es la "variabilidad" que pide la 7.4: dice cuánto cambia el error
       según el periodo que se valide.
     - mape_entrena_media: el MAPE sobre los meses con que se ajustó cada fold. Junto al de
       validación indica subajuste (los dos altos) o sobreajuste (entrenamiento bajo y
       validación alta).
     - mad, msd, rmse: promedios entre folds (contraste del MAPE; RMSE ≫ MAD indica que
       unos pocos meses con errores grandes pesan mucho).
     - mape_ajuste: el MAPE dentro de muestra del modelo reentrenado con todos los meses;
       es la única cifra comparable con el 4,89 % del artículo.
     - mejora_vs_media y mejora_vs_ingenuo: cuánto mejora (en %) el MAPE medio del modelo
       frente a cada línea base del MISMO escenario (mejora_pct de metricas.py).

2. tabla_pareada(): la comparación por fold. Todos los modelos se evalúan en los mismos
   5 folds (mismos meses), así que se puede restar fold por fold:
       dif_k = MAPE del modelo en el fold k − MAPE de la referencia en el fold k
   Negativo = el modelo se equivoca menos que la referencia en ese periodo. Restar pares
   quita la parte del error que se debe a que un periodo es "difícil" para todos, y por eso
   es una comparación más justa que mirar solo las medias. Con 5 folds no hay potencia para
   una prueba de hipótesis seria: se reporta en cuántos folds gana cada uno, sin p-valores.

3. tabla_rango_escenarios(): la segunda fuente de variabilidad que pide la 7.4 (plan de
   cierre, martes 27). Una fila por modelo con su MAPE medio en cada escenario de η, el
   mínimo, el máximo, el rango (máximo − mínimo) y su puesto en cada escenario. Responde
   "¿el resultado depende de cuánta variación regional se simuló?" (pregunta 2 del banco).
"""

import numpy as np
import pandas as pd

from src.config import ESCENARIOS
from src.evaluacion.metricas import mape, mejora_pct

# Nombres de las líneas base tal como quedan en la columna "modelo" de cv_resultados.csv
MEDIA = "media"
INGENUO = "ingenuo_estacional"
# Referencias de la comparación pareada: el ingenuo estacional (la vara que se suponía más
# exigente), la MLR con stepwise (el método del artículo) y, desde el martes 27, la media,
# que resultó ser la línea base más difícil de superar. Con la media aquí, el "gana a la
# media en x de 5 folds" que cita la 7.3 sale de un CSV y no de una cuenta a mano.
REFERENCIAS_PAREADAS = (INGENUO, "mlr_stepwise", MEDIA)

# Columnas mínimas que deben venir en cv_resultados para armar las tablas
COLUMNAS_CV = ["escenario", "modelo", "fold", "hiperparametros", "mape", "mad", "msd",
               "rmse", "mape_entrena", "mape_ajuste"]


def _revisar(cv_resultados: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS_CV if c not in cv_resultados.columns]
    if faltan:
        raise ValueError(f"a cv_resultados le faltan las columnas {faltan}")
    # Cada modelo debe tener exactamente los mismos folds en su escenario: si no, la
    # comparación pareada mezclaría periodos distintos
    folds = cv_resultados.groupby(["escenario", "modelo"])["fold"].apply(
        lambda f: tuple(sorted(f)))
    for escenario, por_modelo in folds.groupby(level="escenario"):
        if por_modelo.nunique() != 1:
            raise ValueError(f"en el escenario {escenario} no todos los modelos tienen "
                             "los mismos folds")


def tabla_comparacion(cv_resultados: pd.DataFrame) -> pd.DataFrame:
    """Una fila por escenario y modelo, ordenada de mejor a peor MAPE medio de validación."""
    _revisar(cv_resultados)
    grupos = cv_resultados.groupby(["escenario", "modelo"], sort=False)
    tabla = grupos.agg(
        mape_media=("mape", "mean"),
        # ddof=1 (desviación muestral): con 5 folds es la estimación usual de la dispersión
        mape_desv=("mape", "std"),
        mape_entrena_media=("mape_entrena", "mean"),
        mad=("mad", "mean"),
        msd=("msd", "mean"),
        rmse=("rmse", "mean"),
        # mape_ajuste e hiperparámetros son uno por modelo (se repiten en cada fold)
        mape_ajuste=("mape_ajuste", "first"),
        hiperparametros=("hiperparametros", "first"),
        n_folds=("fold", "nunique"),
    ).reset_index()

    # Mejora frente a las dos líneas base del mismo escenario (el MAPE medio de cada una)
    for base, columna in [(MEDIA, "mejora_vs_media"), (INGENUO, "mejora_vs_ingenuo")]:
        ref = tabla[tabla["modelo"] == base].set_index("escenario")["mape_media"]
        if ref.empty:
            raise ValueError(f"falta la línea base {base!r} en cv_resultados")
        tabla[columna] = [
            mejora_pct(m, ref[e]) for e, m in zip(tabla["escenario"], tabla["mape_media"])
        ]

    columnas = ["escenario", "modelo", "mape_media", "mape_desv", "mape_entrena_media",
                "mad", "msd", "rmse", "mape_ajuste", "mejora_vs_media", "mejora_vs_ingenuo",
                "hiperparametros", "n_folds"]
    return (tabla[columnas]
            .sort_values(["escenario", "mape_media"])
            .reset_index(drop=True))


def tabla_pareada(cv_resultados: pd.DataFrame,
                  referencias: tuple[str, ...] = REFERENCIAS_PAREADAS) -> pd.DataFrame:
    """Diferencia de MAPE fold por fold de cada modelo frente a cada referencia.

    Una fila por escenario, modelo y referencia, con:
      dif_f1 … dif_f5   MAPE del modelo − MAPE de la referencia en ese fold (puntos %)
      dif_media         promedio de esas diferencias (= diferencia de los MAPE medios)
      gana_en           folds en que el modelo tiene MENOR MAPE que la referencia
      pierde_en         folds en que tiene MAYOR MAPE
    """
    _revisar(cv_resultados)
    anchas = cv_resultados.pivot_table(index=["escenario", "modelo"], columns="fold",
                                       values="mape")
    filas = []
    for (escenario, modelo), mape_modelo in anchas.iterrows():
        for referencia in referencias:
            if modelo == referencia or (escenario, referencia) not in anchas.index:
                continue
            dif = mape_modelo - anchas.loc[(escenario, referencia)]
            fila = {"escenario": escenario, "modelo": modelo, "referencia": referencia}
            fila.update({f"dif_f{k}": v for k, v in dif.items()})
            fila.update({
                "dif_media": dif.mean(),
                "gana_en": int((dif < 0).sum()),
                "pierde_en": int((dif > 0).sum()),
                "n_folds": int(dif.notna().sum()),
            })
            filas.append(fila)
    return (pd.DataFrame(filas)
            .sort_values(["escenario", "referencia", "dif_media"])
            .reset_index(drop=True))


def tabla_rango_escenarios(comparacion: pd.DataFrame) -> pd.DataFrame:
    """Una fila por modelo: MAPE medio en cada escenario, rango entre escenarios y puesto.

    Recibe la salida de tabla_comparacion() (comparacion_cv.csv), no cv_resultados: el
    rango se calcula sobre el MAPE MEDIO de validación de cada escenario, que es la cifra
    con que se ordenan los modelos.

    Columnas:
      mape_<escenario>    MAPE medio de validación en ese escenario (una columna por escenario)
      mape_min, mape_max  el menor y el mayor de esos MAPE
      rango               mape_max − mape_min, en puntos porcentuales. Pequeño = el resultado
                          casi no depende del componente simulado
      puesto_<escenario>  1 = el mejor modelo de ese escenario. Si el puesto cambia entre
                          escenarios, el orden de los modelos depende de la simulación

    Ejemplo: si Ridge tiene 9,02 % en bajo, 9,08 % en central y 9,14 % en alto, su fila
    queda con mape_min = 9,02, mape_max = 9,14 y rango = 0,12 puntos.
    """
    faltan = [c for c in ("escenario", "modelo", "mape_media") if c not in comparacion.columns]
    if faltan:
        raise ValueError(f"a comparacion le faltan las columnas {faltan}")
    # Puesto dentro de cada escenario: el MAPE más bajo es el 1. method="min" da el mismo
    # puesto a dos modelos empatados (no pasa con estas cifras, pero así no se inventa orden)
    puestos = comparacion.assign(
        puesto=comparacion.groupby("escenario")["mape_media"].rank(method="min").astype(int))
    anchas = puestos.pivot(index="modelo", columns="escenario", values=["mape_media", "puesto"])
    # Escenarios en el orden de config (bajo, central, alto); solo los que vengan en la tabla
    escenarios = [e for e in ESCENARIOS if e in anchas["mape_media"].columns]
    mape = anchas["mape_media"][escenarios]
    tabla = mape.add_prefix("mape_")
    tabla["mape_min"] = mape.min(axis=1)
    tabla["mape_max"] = mape.max(axis=1)
    tabla["rango"] = tabla["mape_max"] - tabla["mape_min"]
    tabla = tabla.join(anchas["puesto"][escenarios].astype(int).add_prefix("puesto_"))
    # Orden: como en el escenario central si existe (es el de referencia), si no por el mínimo
    orden = "mape_central" if "mape_central" in tabla.columns else "mape_min"
    tabla.columns.name = None
    return tabla.sort_values(orden).reset_index()


# ---------------------------------------------------------------------------
# Regla para escoger el modelo final (entrega final, lunes 9: congelamiento)
# ---------------------------------------------------------------------------
# Escrita ANTES de abrir la prueba (docs/especificacion_final.md). No se cambia después.
MODELO_POR_DEFECTO = "ridge"
MODELO_RETADOR = "knn"
MESES_MINIMOS_RETADOR = 8  # de 12 meses de prueba


def regla_modelo_final(y_real, pred_ridge, pred_knn, desv_cv_ridge: float,
                       meses_minimos: int = MESES_MINIMOS_RETADOR) -> dict:
    """Aplica la regla congelada sobre la serie NACIONAL de prueba (un valor por mes).

    El modelo final es Ridge. Se cambia a KNN solo si se cumplen LAS DOS condiciones:
      (a) MAPE_ridge − MAPE_knn > desv_cv_ridge, es decir, KNN gana en prueba por más que la
          desviación entre folds del MAPE de validación de Ridge (lo que Ridge "se mueve"
          solo por cambiar de periodo). Desigualdad estricta.
      (b) KNN tiene MENOR error absoluto que Ridge en al menos 8 de los 12 meses. Un empate
          exacto en un mes no cuenta como mes ganado por KNN.
    A igualdad práctica se queda Ridge, por estabilidad e interpretabilidad.

    Ejemplo: Ridge 8,0 %, KNN 7,0 %, desviación de Ridge 0,9 → (a) se cumple (1,0 > 0,9);
    si KNN gana solo 7 meses, (b) no se cumple → el modelo final sigue siendo Ridge.

    Devuelve {"mape_ridge", "mape_knn", "dif_mape", "desv_cv_ridge", "condicion_a",
              "meses_knn_gana", "n_meses", "condicion_b", "modelo_final"}.
    """
    y = np.asarray(y_real, dtype=float)
    r = np.asarray(pred_ridge, dtype=float)
    k = np.asarray(pred_knn, dtype=float)
    if not (y.shape == r.shape == k.shape):
        raise ValueError("observado, Ridge y KNN deben tener los mismos meses")
    mape_r, mape_k = mape(y, r), mape(y, k)
    meses_knn = int((np.abs(y - k) < np.abs(y - r)).sum())
    a = bool(mape_r - mape_k > desv_cv_ridge)
    b = meses_knn >= meses_minimos
    return {
        "mape_ridge": mape_r,
        "mape_knn": mape_k,
        "dif_mape": mape_r - mape_k,
        "desv_cv_ridge": float(desv_cv_ridge),
        "condicion_a": a,
        "meses_knn_gana": meses_knn,
        "n_meses": int(y.size),
        "condicion_b": b,
        "modelo_final": MODELO_RETADOR if (a and b) else MODELO_POR_DEFECTO,
    }


# ---------------------------------------------------------------------------
# Cuánto cuesta ser realistas (entrega final, martes 10, corrección 2)
# ---------------------------------------------------------------------------
MODELOS_COSTO_REZAGOS = ["ridge", "knn", "mlr_stepwise", MEDIA, INGENUO]


def tabla_costo_rezagos(comparacion_rez1: pd.DataFrame, comparacion_reales: pd.DataFrame,
                        escenario: str = "central") -> pd.DataFrame:
    """MAPE de validación con rez1 (Entrega 2) frente a rezagos reales, y su diferencia.

    Recibe las dos comparacion_cv (comparacion_cv_rez1.csv y comparacion_cv.csv, las dos
    corridas con el mismo código) y devuelve una fila por modelo con media ± desviación
    entre folds en cada especificación y diferencia = reales − rez1 en puntos de MAPE.
    Positiva = usar solo lo publicado el día del pronóstico cuesta precisión.

    Ejemplo: Ridge 9,08 % con rez1 y 9,31 % con reales → diferencia +0,23 puntos.
    """
    columnas = ["mape_media", "mape_desv", "hiperparametros"]
    a = comparacion_rez1[comparacion_rez1["escenario"] == escenario].set_index("modelo")
    b = comparacion_reales[comparacion_reales["escenario"] == escenario].set_index("modelo")
    faltan = [m for m in MODELOS_COSTO_REZAGOS if m not in a.index or m not in b.index]
    if faltan:
        raise ValueError(f"faltan los modelos {faltan} en alguna de las dos tablas")
    tabla = pd.DataFrame({"escenario": escenario, "modelo": MODELOS_COSTO_REZAGOS})
    for sufijo, fuente in (("rez1", a), ("reales", b)):
        for col in columnas:
            tabla[f"{col}_{sufijo}"] = fuente.loc[MODELOS_COSTO_REZAGOS, col].to_numpy()
    tabla["diferencia"] = tabla["mape_media_reales"] - tabla["mape_media_rez1"]
    tabla["diferencia_pct"] = 100 * tabla["diferencia"] / tabla["mape_media_rez1"]
    return tabla

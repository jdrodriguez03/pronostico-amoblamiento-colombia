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
"""

import pandas as pd

from src.evaluacion.metricas import mejora_pct

# Nombres de las líneas base tal como quedan en la columna "modelo" de cv_resultados.csv
MEDIA = "media"
INGENUO = "ingenuo_estacional"
# Referencias de la comparación pareada (plan de cierre): la vara exigente y el método
# del artículo
REFERENCIAS_PAREADAS = (INGENUO, "mlr_stepwise")

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

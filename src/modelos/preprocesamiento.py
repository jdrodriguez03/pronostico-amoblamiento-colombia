# Dueño: Juanes
"""ColumnTransformer: codificar categóricas y escalar numéricas. Día 4 de la guía.

Va SIEMPRE dentro del Pipeline (regla de oro 2): se ajusta en cada fold, así que las
categorías del OneHotEncoder y las medias y desviaciones del StandardScaler se aprenden
solo con los meses de entrenamiento de ese fold.

Qué le pasa a cada columna (transformaciones de docs/diccionario_datos.md, propuesta de
Julián del 10 de octubre):

| Columnas                         | Transformación                          |
| -------------------------------- | --------------------------------------- |
| mes, departamento, linea         | one-hot (drop="first"; KNN sin drop)     |
| area_vivienda_m2_rez1            | logaritmo y luego estandarizar           |
| los otros 4 predictores _rez1    | estandarizar                             |
| covid                            | pasa sin cambios (ya es 0/1)             |
| fecha, anio, linea_codigo, ...   | se descartan (remainder="drop")          |
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import PREDICTORES_BIN, PREDICTORES_CAT, predictores_modelo


def columnas_log_por_defecto() -> list[str]:
    """Predictores que llevan logaritmo: el área de vivienda (rezagada o no, según config)."""
    return [c for c in predictores_modelo() if c.startswith("area_vivienda_m2")]


def crear_preprocesador(columnas_log: list[str] | None = None,
                        drop_categorica: str | None = "first") -> ColumnTransformer:
    """OneHotEncoder para mes, departamento y linea; StandardScaler para los numéricos.

    columnas_log: predictores numéricos a los que se aplica logaritmo antes de escalar.
    Por defecto, solo el área de vivienda (docs/diccionario_datos.md). Se deja como
    parámetro para poder probar otra opción con validación cruzada sin editar el código.

    drop_categorica: se pasa tal cual al OneHotEncoder como drop=drop_categorica.
        "first" (por defecto) para los modelos lineales; None para KNN. Por qué:
        - Regresión lineal, Ridge, Lasso: con las 12 dummies de mes MÁS el intercepto, las
          dummies suman 1 en cada fila y son colineales con el intercepto (trampa de las
          variables ficticias). Quitar una categoría (la de referencia) lo evita, y cada
          coeficiente se lee como "diferencia frente a la categoría de referencia".
        - KNN no tiene intercepto ni coeficientes: solo mide distancias. Con drop="first"
          la categoría de referencia queda codificada como (0, 0, ..., 0) y las demás como
          (1, 0, ...), (0, 1, ...). Distancia de la referencia a cualquier otra: 1. Distancia
          entre dos que no son la referencia: √2. Es decir, enero (referencia) quedaría
          "más cerca" de todos los meses que los demás meses entre sí, sin ninguna razón.
          Sin drop, todas las categorías quedan a √2 entre sí: todas son igual de distintas.
    """
    if columnas_log is None:
        columnas_log = columnas_log_por_defecto()
    numericas = predictores_modelo()
    faltan = [c for c in columnas_log if c not in numericas]
    if faltan:
        raise ValueError(f"columnas_log no son predictores del modelo: {faltan}")
    solo_escalar = [c for c in numericas if c not in columnas_log]

    # drop=drop_categorica: "first" quita una categoría de cada variable (la de referencia)
    # para que las dummies no sean colineales con el intercepto; None las deja todas (KNN).
    # handle_unknown="ignore": si en validación aparece una categoría no vista, se codifica
    # con todas sus dummies en 0 en vez de fallar (con "first" eso coincide con la referencia).
    codificador = OneHotEncoder(drop=drop_categorica, handle_unknown="ignore",
                                sparse_output=False)

    # El logaritmo no aprende nada de los datos, pero va aquí dentro para que la misma
    # transformación se aplique igual en entrenamiento, validación y prueba.
    log_y_escalar = Pipeline([
        ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", codificador, PREDICTORES_CAT),
            ("log", log_y_escalar, columnas_log),
            ("num", StandardScaler(), solo_escalar),
            ("bin", "passthrough", PREDICTORES_BIN),
        ],
        remainder="drop",  # fecha, anio, linea_codigo, es_muebles, objetivo y _mismo mes
    )

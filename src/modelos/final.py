# Dueño: Juanes
"""Especificación final del modelo, con los hiperparámetros ya elegidos. Entrega final, martes 10.

Los valores salen de la validación cruzada con la especificación realista (rezagos de
REZAGOS_REALES), escenario central, corrida del martes 10 por la mañana con
`python scripts/entrenar.py --escenario todos --rezagos reales`, ANTES de abrir la prueba.
Están en results/tablas/comparacion_cv.csv (columna hiperparametros):

    ridge  alpha = 10^2,5 ≈ 316,2   (MAPE de validación 9,31 % ± desviación entre folds)
    knn    k = 3, pesos uniformes   (MAPE de validación 9,77 %)

Lo demás (variables, transformaciones, rejillas, agregación, regla) está congelado en
docs/especificacion_final.md, commit 3472249 de main.

Uso (la app de Aleja, src/despliegue/predecir.py):

    from src.modelos.final import ESPECIFICACION_FINAL, construir_modelo_final
    from src.modelos.pipeline import preparar_datos

    datos, _ = preparar_datos(base, ESPECIFICACION_FINAL["rezagos_nombre"])
    modelo = construir_modelo_final().fit(datos, datos["indice_ventas_real"])

construir_modelo_final() devuelve el Pipeline SIN ajustar del modelo final (KNN, k = 3
uniforme); quien lo usa decide con qué meses lo ajusta. construir_modelo_final("ridge") da
el Ridge con alpha = 316,2, que se reporta como contraste.
"""

from src.config import ESCENARIO_PRINCIPAL
from src.features.rezagos import REZAGOS_REALES, columnas_predictoras
from src.modelos.pipeline import candidato_con_predictores, prefijo_parametros

# El modelo final lo decide la regla congelada (regla_modelo_final, comparacion.py) con la
# prueba del escenario central: Ridge salvo que KNN cumpla las dos condiciones. El martes 10
# KNN cumplió las dos (ver docs/decisiones.md, entrada del 10 de noviembre):
#   (a) Ridge 25,73 % − KNN 19,80 % = 5,94 puntos > 5,53 (desviación de Ridge entre folds)
#   (b) KNN tuvo menor error absoluto en 8 de los 12 meses
# Los hiperparámetros NO cambiaron: son los de la validación cruzada, fijados antes de la prueba.
MODELO_FINAL = "knn"

HIPERPARAMETROS = {
    # np.logspace(-3, 5, 17) tiene pasos de media década: el elegido es 10 ** 2,5
    "ridge": {"alpha": 10 ** 2.5},
    "knn": {"n_neighbors": 3, "weights": "uniform"},
}

ESPECIFICACION_FINAL = {
    "modelo": MODELO_FINAL,
    "hiperparametros": HIPERPARAMETROS,
    "rezagos": dict(REZAGOS_REALES),
    "rezagos_nombre": "reales",          # el argumento de preparar_datos y --rezagos
    "predictores": columnas_predictoras(),
    "escenario": ESCENARIO_PRINCIPAL,    # central
    "commit_congelamiento": "34722497952b2a8d23ec7e4d18c23e7de3039517",
}


def construir_modelo_final(nombre: str = MODELO_FINAL):
    """Pipeline sin ajustar de Ridge o KNN con los hiperparámetros fijos y los rezagos reales.

    Es el mismo Pipeline que evaluó GridSearchCV (candidato_con_predictores), con el
    valor ganador puesto a mano: objetivo en logaritmo, one-hot (sin drop para KNN),
    StandardScaler y el modelo.
    """
    if nombre not in HIPERPARAMETROS:
        raise ValueError(f"nombre debe ser uno de {sorted(HIPERPARAMETROS)}, no {nombre!r}")
    pipeline, _ = candidato_con_predictores(nombre, columnas_predictoras())
    valores = {f"{prefijo_parametros()}{clave}": valor
               for clave, valor in HIPERPARAMETROS[nombre].items()}
    return pipeline.set_params(**valores)

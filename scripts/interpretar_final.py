# Dueño: Julián
"""Interpretación del modelo final (entrega final). Solo usa entrenamiento (feb 2019 – jul 2025).

Guarda:
- results/tablas/coeficientes_final.csv: coeficientes estandarizados de Ridge (contraste
  interpretable) con media ± desviación entre los 5 folds y el ajuste con todo el periodo.
- results/tablas/vif_final.csv: VIF de los predictores con rezagos reales.
- results/tablas/knn_vecinos.csv: de dónde salen los 3 vecinos de KNN (misma serie, mismo
  mes calendario, año medio de los vecinos por año de la fila).

Uso: python scripts/interpretar_final.py
"""

import numpy as np
import pandas as pd

from src.config import OBJETIVO, RUTA_TABLAS
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.interpretacion import (
    calcular_vif,
    coeficientes,
    coeficientes_por_fold,
    estabilidad,
)
from src.evaluacion.particion import CVTemporalPorMes
from src.features.rezagos import columnas_predictoras
from src.modelos.final import construir_modelo_final
from src.modelos.pipeline import preparar_datos


def main() -> None:
    datos, _ = preparar_datos(cargar_entrenamiento("central"), "reales")
    y = datos[OBJETIVO]

    ridge = construir_modelo_final("ridge").fit(datos, y)
    total = coeficientes(ridge).set_index("variable")["coeficiente"].rename("coef_todo")
    por_fold = coeficientes_por_fold(construir_modelo_final("ridge"), datos,
                                     CVTemporalPorMes(), OBJETIVO)
    tabla = estabilidad(por_fold).set_index("variable").join(total).reset_index()
    tabla = tabla[tabla["variable"].str.contains("_rez|covid")]
    tabla["efecto_pct_1desv"] = (np.exp(tabla["media"]) - 1) * 100
    tabla.to_csv(RUTA_TABLAS / "coeficientes_final.csv", index=False)

    calcular_vif(datos, columnas_predictoras()).to_csv(RUTA_TABLAS / "vif_final.csv", index=False)

    knn = construir_modelo_final("knn").fit(datos, y)
    pipe = knn.regressor_
    vecinos = pipe[-1].kneighbors(pipe[:-1].transform(datos), n_neighbors=4,
                                  return_distance=False)[:, 1:]
    serie = (datos["departamento"].astype(str) + "|" + datos["linea_codigo"].astype(str)).to_numpy()
    fecha = pd.to_datetime(datos["fecha"])
    mes, anio = fecha.dt.month.to_numpy(), fecha.dt.year.to_numpy()
    resumen = pd.DataFrame({
        "anio_fila": anio,
        "misma_serie": (serie[vecinos] == serie[:, None]).mean(axis=1),
        "mismo_mes": (mes[vecinos] == mes[:, None]).mean(axis=1),
        "anio_vecinos": anio[vecinos].mean(axis=1),
    }).groupby("anio_fila").mean().round(3).reset_index()
    resumen.to_csv(RUTA_TABLAS / "knn_vecinos.csv", index=False)
    print(tabla.round(4).to_string(index=False))
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()

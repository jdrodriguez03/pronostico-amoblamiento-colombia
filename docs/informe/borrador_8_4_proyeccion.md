# 8.4 Limitaciones del modelamiento y proyección (borrador)

Dueño: Julián · Media página · Complementa la parte de datos y simulación de Aleja · Cifras de `comparacion_cv_rezagos.csv` y `tabla_final.csv`.

**Limitaciones.**

- **Disponibilidad de los datos.** Usar solo lo publicado el día de emisión cuesta 1,01 puntos de MAPE de validación en KNN (8,76 % → 9,77 %) y 0,23 en Ridge. La tasa de consumo entra con un mes de rezago como promedio de las semanas ya publicadas: es un supuesto, porque el primer día del mes puede faltar la última semana.
- **Pronóstico ex post.** En la prueba los predictores son los valores observados de esos meses, no pronósticos de ellos. Con predictores pronosticados el error sería mayor.
- **Optimismo de la validación.** El MAPE de validación del modelo elegido (9,77 %) se obtuvo con los mismos folds que sirvieron para elegirlo; la cifra honesta es la de prueba (19,80 %).
- **Doce meses de prueba.** Los intervalos por bloques son anchos (16,5 % a 22,8 % para KNN) y la diferencia frente al ingenuo estacional no es concluyente (−0,27 a 7,47 puntos).
- **Extrapolación.** KNN no puede pronosticar por encima de lo ya observado y Ridge tampoco anticipa el crecimiento de 2024–2026: ninguno de los dos sigue un cambio de nivel del mercado.

**Mejoras y proyección.**

- Usar el ingenuo estacional como referencia obligatoria y modelar el cambio frente al mismo mes del año anterior (crecimiento anual) en lugar del nivel, para que el modelo pueda acompañar un mercado que crece. Esta variante se dejaría especificada y elegida con validación cruzada, y se evaluaría con los meses posteriores a julio de 2026, a medida que el DANE los publique: una prueba nueva, que nadie ha visto.
- Reentrenar cada mes con ventana móvil, para incorporar el nivel reciente.
- Conseguir datos reales por departamento y datos de unidades, inventario y precios, para pasar de una señal de índice a una decisión de inventario.
- En Machine Learning II: árboles y ensambles (random forest, boosting) para relaciones no lineales e interacciones, y modelos de series de tiempo con variables exógenas para la tendencia y la estacionalidad.

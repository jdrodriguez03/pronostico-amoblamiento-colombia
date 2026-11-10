# 8.1 Modelo final y evaluación en prueba (borrador)

Dueño: Juanes · Máximo una página en el informe · Cifras de `results/tablas/tabla_final.csv`, `prueba.csv`, `prueba_pareada.csv`, `comparacion_cv.csv` y `comparacion_cv_rezagos.csv` (corridas del 10 de noviembre). Figura: `results/figuras/fig_prueba_serie_nacional.png`.

**Configuración definitiva.** El modelo pronostica el índice de ventas reales (2019 = 100) del mes t por departamento × línea, no unidades, pesos ni inventario. Usa solo lo publicado el día de emisión: tasa de consumo con un mes de rezago, ICC e inflación con dos, importaciones y licencias con tres. El objetivo entra en logaritmo; las categorías van en one-hot (sin `drop` en KNN) y los numéricos se estandarizan dentro del Pipeline. Los hiperparámetros los eligió la validación cruzada antes de abrir la prueba: Ridge con α = 316 y KNN con k = 3 y pesos uniformes. Todo quedó congelado en `docs/especificacion_final.md` (commit `3472249`). Usar solo información disponible cuesta 0,23 puntos de MAPE de validación en Ridge (9,08 % → 9,31 %) y 1,01 puntos en KNN (8,76 % → 9,77 %).

**Por qué KNN.** La regla se escribió antes de abrir la prueba: Ridge, salvo que KNN gane en prueba por más de la desviación entre folds de Ridge y tenga menor error en al menos 8 de los 12 meses. En el escenario central KNN cumple las dos condiciones: 19,80 % frente a 25,73 %, una diferencia de 5,94 puntos frente a una desviación de 5,53, y menor error en 8 de 12 meses. El resultado es el mismo en los tres escenarios. Las dos condiciones se cumplen por poco (0,40 puntos de holgura en la primera y un mes ganado por 0,14 puntos del índice en la segunda), y el IC 95 % de la diferencia Ridge − KNN (−1,35 a 12,03) cruza el cero. Se elige KNN por la regla, no porque la diferencia sea concluyente. Ridge sigue siendo el modelo interpretable y el más estable en validación (9,31 % frente a 9,77 %).

**Resultados en prueba (agosto 2025 – julio 2026, una sola evaluación).**

| Modelo | MAPE validación (5 folds) | MAPE prueba | IC 95 % por bloques | Prueba − ingenuo (IC 95 %) |
| --- | --- | --- | --- | --- |
| KNN (final) | 9,77 ± 4,17 % | 19,80 % | 16,53 – 22,82 % | +3,74 (−0,27 a 7,47) |
| Ridge | 9,31 ± 5,53 % | 25,73 % | 18,63 – 31,58 % | +9,68 (2,29 a 15,34) |
| MLR con stepwise | 12,26 ± 4,93 % | 29,13 % | 17,26 – 38,93 % | +13,07 (1,21 a 22,51) |
| Media por serie | 9,88 ± 5,44 % | 27,56 % | 24,04 – 29,95 % | +11,50 (6,89 a 14,75) |
| Ingenuo estacional | 12,30 ± 7,76 % | 16,06 % | 14,85 – 17,58 % | — |

El IC se calcula con bootstrap de bloques de 3 meses consecutivos sobre la serie nacional (B = 2000), nunca fila por fila. Con 12 meses los intervalos salen anchos.

**Cuánto mejora frente a no tener modelo.** Frente a la media de cada serie, KNN reduce el MAPE de prueba un 28 % (27,56 % → 19,80 %). **Frente al ingenuo estacional no mejora**: es 3,74 puntos peor, y el IC de esa diferencia (−0,27 a 7,47) cruza el cero, así que tampoco se puede afirmar que sea peor. Ridge sí es peor que el ingenuo con un IC que no cruza el cero. En validación, en cambio, todos los modelos le ganaban al ingenuo: KNN un 21 % y Ridge un 24 %.

**Dónde se equivoca más.** Todos los modelos, incluido el ingenuo, subestiman la serie nacional en los 12 meses: con KNN, lo observado supera en promedio en un 26 % a lo predicho. Los mayores errores de KNN están en enero de 2026 (30 %), noviembre de 2025 (28 %, mes de mayor venta), marzo de 2026 (27 %) y mayo de 2026 (27 %); los menores, en febrero y abril de 2026 (6 % y 8 %). Por línea, el error del panel se concentra en sonido y video (37 %), informática (36 %) y prendas de vestir (32 %), frente a 19 % en muebles. Es el patrón de subestimación de 2024–2025 (8.4): la serie creció y los predictores rezagados no lo anticipan. Por la regla 3, el modelo no se modificó por esto.

**Lectura honesta.** El MAPE de validación del modelo elegido (9,77 %) es optimista, porque se eligió con esos mismos folds. La cifra honesta es la de la prueba: 19,80 % (IC 16,5 a 22,8 %). La caída de validación a prueba, mayor que la variación entre escenarios (0,5 puntos), indica que el problema es el cambio de nivel de la serie y no la parte simulada.

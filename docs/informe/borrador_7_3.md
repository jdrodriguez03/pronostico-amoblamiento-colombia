# 7.3 Entrenamiento y comparación de modelos (borrador)

Dueño: Julián · Máximo una página en el informe · Los huecos `[pendiente: ...]` se llenan con
`results/tablas/comparacion_cv.csv` después de la corrida oficial (días 9 a 11).

## Modelos comparados

Se compararon cinco modelos dentro del mismo Pipeline (codificación one-hot de mes, departamento y
línea; logaritmo del área de vivienda; estandarización de los predictores rezagados), todos sobre el
logaritmo del índice de ventas:

| Modelo | Por qué se incluye |
| --- | --- |
| MLR con stepwise (α = 0,25) | Reproduce el método del artículo base |
| MLR completa | Muestra el efecto de usar todos los predictores sin selección |
| Ridge | Encoge los coeficientes de predictores correlacionados para estabilizarlos |
| Lasso | Encoge y anula coeficientes: selecciona variables por validación, no por p-valores |
| KNN para regresión | Contraste no lineal |

Se contrastan con dos líneas base: el predictor de la media de cada serie y el ingenuo estacional
(el valor de la misma serie doce meses antes), que es la vara más exigente porque ya recoge la
estacionalidad.

## Ajuste de hiperparámetros

Los hiperparámetros se eligieron con `GridSearchCV` y validación cruzada temporal por mes
(`CVTemporalPorMes`, 5 folds de ventana creciente, febrero de 2019 a julio de 2025), minimizando el
MAPE de la serie nacional de muebles. Todo lo que aprende de los datos, incluida la selección
stepwise, se ajusta dentro de cada fold.

| Modelo | Rejilla | Elegido |
| --- | --- | --- |
| MLR con stepwise | α fijo en 0,25 (decisión del artículo, no se ajusta) | — |
| Ridge | alpha en 13 valores de 0,001 a 1000 (escala log) | [pendiente] |
| Lasso | alpha en 11 valores de 0,0001 a 10 (escala log) | [pendiente] |
| KNN | k en {3, 5, 10, 20, 40}; pesos uniformes o por distancia | [pendiente] |

[pendiente: una frase por curva de validación (figuras del día 9): dónde está el mínimo y qué tan
plana es la curva alrededor de él.]

## Resultados de la validación cruzada (escenario central)

| Modelo | MAPE nacional (media ± desv. entre folds) | Mejora vs. ingenuo estacional |
| --- | --- | --- |
| MLR con stepwise | [pendiente] | [pendiente] |
| MLR completa | [pendiente] | [pendiente] |
| Ridge | [pendiente] | [pendiente] |
| Lasso | [pendiente] | [pendiente] |
| KNN | [pendiente] | [pendiente] |
| Ingenuo estacional | [pendiente] | — |
| Predictor de la media | [pendiente] | [pendiente] |

[pendiente: lectura en dos frases: qué modelos superan al ingenuo estacional y si el orden cambia en
los escenarios bajo y alto (análisis de Aleja, día 10).]

## Artículo frente a este trabajo

| Aspecto | Artículo base (EE. UU., 2019–2023) | Este trabajo (Colombia, 2019–2025) |
| --- | --- | --- |
| Método | MLR con stepwise (Minitab, α = 0,25), 60 meses con ajuste estacional | 5 modelos con validación cruzada temporal por mes; estacionalidad con dummies |
| MAPE de ajuste (mismos meses con que se estima el modelo) | 4,89 % (MLR); 6 % (Holt–Winters), sobre 2019–2023 | [pendiente: MAPE nacional de ajuste del modelo final, reentrenado con feb 2019 – jul 2025] |
| MAPE fuera de muestra (meses no vistos) | No lo reporta | [pendiente: validación cruzada, media ± desv.; prueba ago 2025 – jul 2026] |
| Variables retenidas | Importaciones, sentimiento del consumidor, inicios de vivienda, población, precios al productor, pandemia | [pendiente: variables elegidas por el stepwise y no anuladas por Lasso] |
| Signos | Población negativa (−0,265), contraria a la teoría | [pendiente: signo de cada coeficiente frente al esperado] |
| VIF máximo | 23,18 (precios al productor); 16,16 (población) | 6,28 (inflación), ninguno sobre 10 |
| R² de ajuste | 76,73 % (ajustado 74,09 %) | [pendiente, si se reporta: R² de ajuste del modelo final] |

Signos esperados según la teoría: importaciones (+), ICC (+), área de vivienda (+), inflación (−),
tasa de consumo (−), covid (−).

**Qué cifra se compara con cuál.** El artículo mide su error sobre los mismos meses con que
estima el modelo: para Holt–Winters lo declara como métrica dentro de muestra (p. 15), y el MAPE
de la MLR aparece en la tabla de pronósticos de 2024 a 2026, meses sin valores observados, así
que solo pudo calcularse sobre los meses de ajuste (2019–2023). Por eso el 4,89 % se compara con
el MAPE de ajuste de nuestro modelo final, no con el de prueba. El MAPE de validación y de prueba
responde otra pregunta, cuánto se equivoca el modelo en meses que no vio, y el artículo no lo
reporta. Se espera que sea mayor que el de ajuste sin que eso signifique un modelo peor.

Texto propuesto para el informe: *"El artículo reporta el error sobre los mismos meses con que
ajusta el modelo (4,89 %). Nuestro modelo, medido de la misma forma, obtiene [X] %. Además
reportamos el error en meses no vistos, [Y] % en validación cruzada y [Z] % en prueba, que es la
medida de capacidad predictiva."*

Nota: el resumen del artículo reporta 3,47 % y 4,21 %, cifras que no coinciden con su tabla de
resultados (4,89 % y 6 %); se usa la de la tabla (ver `docs/decisiones.md`).

## Respuesta a la pregunta de la Entrega 1

[pendiente: ¿la regularización mejora la capacidad predictiva frente a la MLR cuando hay predictores
correlacionados? Una frase con el MAPE de Ridge y Lasso frente a la MLR, y otra con la estabilidad
de los coeficientes (`estabilidad()`, media ± desviación entre folds).]
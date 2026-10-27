# 7.3 Entrenamiento y comparación de modelos (borrador)

Dueño: Julián · Máximo una página en el informe · Cifras preliminares de
`notebooks/04_interpretacion.ipynb` (escenario central, solo entrenamiento). Se confirman contra
`results/tablas/comparacion_cv.csv` cuando corra `scripts/entrenar.py`; si coinciden, se
quita esta nota.

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
| KNN para regresión | Contraste no lineal; usa el one-hot sin `drop` para no sesgar las distancias |

Se contrastan con dos líneas base: el predictor de la media de cada serie y el ingenuo estacional
(el valor de la misma serie doce meses antes). Se esperaba que el ingenuo fuera la más exigente,
porque ya recoge la estacionalidad, pero en la validación resultó serlo la media (ver abajo).

## Ajuste de hiperparámetros

Los hiperparámetros se eligieron con `GridSearchCV` y validación cruzada temporal por mes
(`CVTemporalPorMes`, 5 folds de ventana creciente, febrero de 2019 a julio de 2025), minimizando el
MAPE de la serie nacional de muebles. Todo lo que aprende de los datos, incluida la selección
stepwise, se ajusta dentro de cada fold.

| Modelo | Rejilla | Elegido | MAPE de validación |
| --- | --- | --- | --- |
| MLR con stepwise | α fijo en 0,25 (decisión del artículo, no se ajusta) | — | 15,0 % |
| Ridge | alpha en 17 valores de 0,001 a 100.000 (escala log) | alpha = 1000 | 9,1 % |
| Lasso | alpha en 11 valores de 0,0001 a 10 (escala log) | alpha ≈ 0,032 | 9,4 % |
| KNN | k en {1, 2, 3, 5, 10, 20, 40}; pesos uniformes o por distancia | k = 3, uniformes | 9,0 % |

Las rejillas de Ridge y KNN se ampliaron porque el mínimo caía en su borde (alpha = 1000 y k = 3);
con la ampliación las dos curvas vuelven a subir, así que son mínimos reales (figura de curvas de
validación). Ningún valor más simple es equivalente al mínimo según la comparación pareada fold a
fold; en Ridge, alpha = 3162 queda muy cerca (9,6 %). KNN con k = 1 o con pesos por distancia
tiene error de entrenamiento 0 %: memoriza, y por eso pierde en validación. Con regularización
fuerte el error de entrenamiento (≈ 13 %) supera al de validación (≈ 9 %) porque solo el
entrenamiento incluye los meses de pandemia.

## Resultados de la validación cruzada (escenario central)

| Modelo | MAPE nacional (media ± desv. entre folds) | Mejora vs. ingenuo estacional | Mejora vs. media | Folds en que gana a la media |
| --- | --- | --- | --- | --- |
| KNN | 9,0 % ± 4,5 | 26,8 % | 12,2 % | 3 de 5 |
| Ridge | 9,1 % ± 6,0 | 26,2 % | 11,5 % | 3 de 5 |
| Lasso | 9,4 % ± 5,7 | 23,6 % | 8,3 % | 4 de 5 |
| Predictor de la media | 10,3 % ± 5,9 | 16,6 % | — | — |
| Ingenuo estacional | 12,3 % ± 7,8 | — | −19,9 % | 2 de 5 |
| MLR completa | 14,8 % ± 6,5 | −20,3 % | −44,3 % | 1 de 5 |
| MLR con stepwise | 15,0 % ± 6,7 | −21,8 % | −46,0 % | 1 de 5 |

Los tres modelos regularizados o no lineales superan a las dos líneas base en promedio, pero la
ventaja es modesta frente a la media (8 % a 12 %) y no ocurre en todos los folds: Lasso es el más
consistente (gana a las dos líneas base en 4 de 5 folds). La MLR, con o sin stepwise, queda por
debajo de las dos líneas base. [pendiente: si el orden cambia en los escenarios bajo y alto
(análisis de Aleja).]

## Artículo frente a este trabajo

| Aspecto | Artículo base (EE. UU., 2019–2023) | Este trabajo (Colombia, 2019–2025) |
| --- | --- | --- |
| Método | MLR con stepwise (Minitab, α = 0,25), 60 meses con ajuste estacional | 5 modelos con validación cruzada temporal por mes; estacionalidad con dummies |
| MAPE de ajuste (mismos meses con que se estima el modelo) | 4,89 % (MLR); 6 % (Holt–Winters), sobre 2019–2023 | MLR con stepwise 12,8 %; Ridge 13,0 % (sin abril–agosto de 2020: 11,4 % y 11,3 %) |
| MAPE fuera de muestra (meses no vistos) | No lo reporta | Validación cruzada: MLR con stepwise 15,0 % ± 6,7; Ridge 9,1 % ± 6,0. La prueba (ago 2025 – jul 2026) queda para la entrega final |
| Variables retenidas | Importaciones, sentimiento del consumidor, inicios de vivienda, población, precios al productor, pandemia | Stepwise: las 6 (covid, importaciones, ICC, inflación, tasa de consumo, vivienda), todas con p < 1e-4. Lasso: importaciones e inflación; anula ICC, tasa de consumo y covid |
| Signos | Población negativa (−0,265), contraria a la teoría | Stepwise: importaciones, tasa y covid con el signo esperado; ICC, inflación y vivienda al revés. Ridge: solo ICC e inflación al revés, cerca de cero |
| VIF máximo | 23,18 (precios al productor); 16,16 (población) | 6,28 (inflación), ninguno sobre 10 |
| R² de ajuste | 76,73 % (ajustado 74,09 %) | No se reporta: el artículo lo calcula sobre una serie desestacionalizada y aquí la serie conserva su estacionalidad, así que no es comparable |

Signos esperados según la teoría: importaciones (+), ICC (+), área de vivienda (+), inflación (−),
tasa de consumo (−), covid (−).

**Qué cifra se compara con cuál.** El artículo mide su error sobre los mismos meses con que
estima el modelo: para Holt–Winters lo declara como métrica dentro de muestra (p. 15), y el MAPE
de la MLR aparece en la tabla de pronósticos de 2024 a 2026, meses sin valores observados, así
que solo pudo calcularse sobre los meses de ajuste (2019–2023). Por eso el 4,89 % se compara con
nuestro MAPE de ajuste, no con el de validación. Nuestro error de ajuste es mayor (12,8 % a
13,0 %) por dos razones: incluye los meses de pandemia, donde el índice cae a la mitad, y un mismo
modelo se ajusta a las 56 series del panel, mientras el artículo ajusta una sola serie. El MAPE
de validación responde otra pregunta, cuánto se equivoca el modelo en meses que no vio, y el
artículo no lo reporta.

Texto propuesto para el informe: *"El artículo reporta el error sobre los mismos meses con que
ajusta el modelo (4,89 %). Medido de la misma forma, su método (MLR con stepwise) obtiene aquí
12,8 %, en buena parte por los meses de pandemia (11,4 % sin ellos). Más importante, en meses no
vistos la MLR con stepwise se equivoca en 15,0 %, más que el simple promedio de cada serie
(10,3 %), mientras Ridge y Lasso bajan a 9,1 % y 9,4 %."*

Nota: el resumen del artículo reporta 3,47 % y 4,21 %, cifras que no coinciden con su tabla de
resultados (4,89 % y 6 %); se usa la de la tabla (ver `docs/decisiones.md`).

## Respuesta a la pregunta de la Entrega 1

¿La regularización mejora la capacidad predictiva frente a la MLR cuando hay predictores
correlacionados? Sí. Ridge y Lasso bajan el MAPE nacional de validación de 14,8 % (MLR completa)
a 9,1 % y 9,4 %, y estabilizan los coeficientes: en la MLR completa el área de vivienda cambia de
signo entre folds, mientras con Ridge ningún coeficiente cambia de signo y la desviación entre
folds es entre 1,6 y 25 veces menor (figura de estabilidad de coeficientes).

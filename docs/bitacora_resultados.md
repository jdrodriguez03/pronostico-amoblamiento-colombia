# Bitácora de resultados

Dueño: Julián

Fecha, qué se corrió y qué salió.


## 2026-10-15 · Exploración de los candidatos, escenario central (antes de `entrenar.py`)

Corrida exploratoria para comprobar que los 5 candidatos entrenan en el Pipeline de Juanes con
`CVTemporalPorMes` (5 folds) y `GridSearchCV`. **No es la corrida oficial:** usa el MAPE del panel
(scoring de scikit-learn), no el MAPE nacional de la línea 8, que llega con `scorer_mape_nacional`
de Juanes y `agregar_nacional` de Aleja. Solo entrenamiento (feb-2019 a jul-2025).

| Modelo | Hiperparámetro elegido | MAPE panel CV |
| --- | --- | --- |
| mlr_stepwise | α = 0,25 (fijo) | 17,6 % |
| mlr_completa | — | 17,4 % |
| ridge | alpha = 316 | 15,9 % |
| lasso | alpha = 0,01 | 17,0 % |
| knn | k = 5, distance | 16,1 % |

Primeras lecturas (preliminares):

- **Stepwise:** en la base completa entran los 6 (covid, importaciones, tasa de consumo, ICC,
  inflación y vivienda), todos con p < 1e-4: confirma que los p-valores son optimistas. Entre folds,
  vivienda solo queda en 3 de 5 e importaciones en 4 de 5.
- **Estabilidad de coeficientes (media ± desviación entre folds):** en la MLR completa el área de
  vivienda cambia de signo entre folds y el ICC sale negativo (−0,061 ± 0,011), contra lo que dice la
  teoría. Con Ridge (alpha = 316) vivienda queda positiva en los 5 folds (0,018 ± 0,003): primera
  evidencia de que la regularización estabiliza los coeficientes.
- Ridge y KNN superan a la MLR en el panel. Falta compararlos con el ingenuo estacional y en la
  escala nacional antes de concluir.

## 2026-10-25 · Lectura de las verificaciones de la simulación

- **Correlación entre líneas:** las medias de correlación de cambios son 0,646, 0,594 y 0,496
  en los escenarios bajo, central y alto, respectivamente; las tres superan el rango descriptivo
  de 0,07–0,30. No hay en las fuentes del proyecto series DANE comparables por departamento y
  línea para validar ese rango empíricamente, así que se reporta como una limitación de la
  simulación, sin cambiar el criterio.
- **Participaciones departamentales:** las 21 comparaciones (7 departamentos por 3 escenarios)
  quedan fuera de la tolerancia de igualdad de 1e-6; la diferencia absoluta máxima es 0,0136188
  (1,36 puntos porcentuales), en Bogotá para el escenario bajo. `PESOS_DEPTO` son coeficientes
  para agregar los índices, no participaciones observadas fijas que cada escenario deba reproducir.

## 2026-10-26 · Curvas de validación y estabilidad de coeficientes, escenario central

Corrida con `notebooks/04_interpretacion.ipynb`: mismo Pipeline, misma validación cruzada por
mes y mismo `scorer_mape_nacional` que usará `scripts/entrenar.py`, solo con entrenamiento. Se
debe comprobar contra `cv_resultados.csv` cuando exista (última sección del cuaderno).

| Modelo | Elegido | MAPE nacional de validación |
| --- | --- | --- |
| Ridge | alpha = 1000 | 9,1 % ± 6,0 |
| Lasso | alpha ≈ 0,032 | 9,4 % ± 5,7 |
| KNN | k = 3, pesos uniformes | 9,0 % ± 4,5 |
| Ridge con alpha casi cero (≈ MLR completa) | — | 14,8 % |

- **Bordes de rejilla.** Ridge (alpha = 1000) y KNN (k = 3) caían en el borde de sus rejillas
  anteriores. Se ampliaron (Ridge hasta 1e5, KNN desde k = 1) y en los dos casos la curva vuelve
  a subir: son mínimos reales.
- **Parsimonia.** Con la comparación pareada fold a fold (regla de un error estándar sobre las
  diferencias), ningún valor más simple es equivalente al mínimo en los tres modelos. En Ridge,
  3162 queda muy cerca (Δ 0,48 frente a un error estándar de 0,43).
- **Sobreajuste.** KNN con pesos por distancia tiene MAPE de entrenamiento 0 para todo k, y con
  k = 1 llega a 16,2 % en validación: memoriza. Con regularización fuerte el entrenamiento
  (≈ 13 %) supera a la validación (≈ 9 %) porque solo el entrenamiento incluye los meses de
  pandemia.
- **Estabilidad.** En la MLR completa el área de vivienda cambia de signo entre folds; con Ridge
  ningún coeficiente cambia de signo y la desviación entre folds es entre 1,6 y 25 veces menor.
- **KNN reproducible.** En cerca de un tercio de las filas el k-ésimo vecino empata en distancia
  con el siguiente, y con el algoritmo por defecto el MAPE con k = 3 salía 9,23 % o 8,99 % según
  se corriera en paralelo o no. Con `algorithm="kd_tree"` da 9,00 % siempre.

## 2026-10-26 · Comparación con el artículo y con las líneas base, escenario central

Mismo cuaderno, sección 7 (`results/tablas/comparacion_articulo.csv` y `stepwise_seleccion.csv`).

- **El predictor de la media es la línea base más exigente**, no el ingenuo estacional: 10,3 %
  frente a 12,3 %, y le gana en 3 de 5 folds. El ingenuo falla sobre todo en el fold 1 (21,3 %).
- **Ridge, Lasso y KNN superan a las dos líneas base en promedio, pero no en todos los folds.**
  Frente a la media mejoran entre 8 % y 12 %; Ridge y KNN le ganan en 3 de 5 folds y Lasso en 4.
  Frente al ingenuo, Ridge solo gana en 2 de 5: su ventaja media (26 %) descansa en el fold 1.
- **El método del artículo no supera a ninguna línea base**: la MLR con stepwise queda en 15,0 %.
  Retiene las 6 variables con p < 1e-4, y ICC, inflación y vivienda salen con el signo contrario.
- **MAPE de ajuste** (comparable con el 4,89 % del artículo): 12,8 % la MLR con stepwise y 13,0 %
  Ridge; sin abril a agosto de 2020, 11,4 % y 11,3 %. Es mayor que el de validación porque el
  ajuste incluye la pandemia.
- **Descartado:** dar a cada línea su propia estacionalidad (dummies mes × línea) no mejora a
  Ridge (9,23 % frente a 9,08 %).

## 2026-10-26 · Comparación entre escenarios

El orden por MAPE nacional se mantiene entre bajo y central, pero cambia en alto: Ridge queda primero, KNN pasa del primer puesto al cuarto (10,261 %, apenas sobre la media con 10,257 %) y MLR stepwise supera por poco a MLR completa. El MAPE nacional varía poco porque la serie observada es la misma y la simulación conserva su agregado, aunque no es invariable: los modelos se ajustan sobre paneles distintos. El MAPE del panel sí cambia más al aumentar σ; por ejemplo, para la línea base de la media pasa de 13,204 % a 14,685 % y 17,255 % en bajo, central y alto (`results/tablas/escenarios.csv`).

## 2026-10-26 · Estabilidad de la selección

La selección cambia entre folds o escenarios para: importaciones_muebles_usd_miles_rez1 (Stepwise 14/15; Lasso 15/15), area_vivienda_m2_rez1 (Stepwise 6/15; Lasso 9/15).

## 2026-10-27 · Análisis de errores del modelo preliminar

El equipo eligió Ridge (alpha = 1000) como modelo preliminar: en central KNN queda 0,09 puntos porcentuales por debajo de Ridge, diferencia pequeña frente al error estándar pareado informado (0,87 puntos); se priorizaron la estabilidad de Ridge entre escenarios (9,02–9,14 %) y sus coeficientes sin cambios de signo. Los residuos de validación se recalcularon con el pipeline actual. Los residuos son observado menos predicho: positivos indican subestimación y negativos, sobreestimación. Los tres mayores errores absolutos fueron Jul 2025: +46.93 puntos; May 2025: +32.85 puntos; Nov 2024: +29.21 puntos. El error absoluto medio por mes calendario fue mayor en julio (23.09 puntos; solo 2 o 3 observaciones por mes), por lo que no basta para afirmar un patrón estacional recurrente. En ajuste dentro de muestra, los cinco meses con `covid=1` tuvieron MAPE 37.85 % y MAD 27.59, frente a 11.28 % y 14.01 en los demás meses. Son residuos dentro de muestra, no errores de pronóstico. Los meses se conservaron y se identifican con `covid`; no se calculó distancia de Cook. El MAPE medio recalculado (9.079339 %) coincide con el guardado (9.079339 %).

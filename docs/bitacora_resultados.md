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
| KNN | k = 3, pesos uniformes | 9,0 % ± 4,6 |
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

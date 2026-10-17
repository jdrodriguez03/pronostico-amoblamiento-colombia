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
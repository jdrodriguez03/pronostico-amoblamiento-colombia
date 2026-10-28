# 7.4 Validación, métricas y variabilidad (borrador)

Dueño: Juanes · Máximo media página en el informe · Cifras de `results/tablas/comparacion_cv.csv`,
`comparacion_pareada.csv`, `rango_escenarios.csv` y `cv_resultados.csv` (corrida de
`scripts/entrenar.py --escenario todos` del 27 de octubre). Cada cifra está rastreada en
`docs/cuadre_cifras.md`.

**Estrategia de validación.** Validación cruzada temporal por mes (`CVTemporalPorMes`): 5 folds de
ventana creciente sobre febrero de 2019 – julio de 2025; cada fold entrena con todos los meses
anteriores (48 a 72) y valida los 6 siguientes, para 30 meses validados de febrero de 2023 a julio
de 2025. Las 56 filas de un mes quedan siempre del mismo lado. La prueba (agosto de 2025 – julio de
2026) no se usa en esta entrega.

**Métricas.** Se mide sobre la serie nacional de muebles, en la escala del índice. El **MAPE** es la
métrica principal: es la del artículo (4,89 %), no depende de la escala y se lee directo como "me
equivoco en x % de las ventas", que es lo que importa para pedir inventario. Su debilidad es que se
dispara cuando el valor observado es bajo, como en los meses de 2020; por eso se contrasta con el
**MAD** (error medio en puntos del índice, que no depende del nivel) y con el **RMSE**, que castiga
los errores grandes: si RMSE ≫ MAD, unos pocos meses concentran el error. Los hiperparámetros se
eligen por el MAPE nacional (`scorer_mape_nacional`).

**Tabla (numeración provisional). Resultados de la validación cruzada, escenario central.**

| Modelo | MAPE (media ± desv. entre folds) | MAD | RMSE | Mejora vs. media | Mejora vs. ingenuo | Folds en que gana al ingenuo | MAPE en los 3 escenarios |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN (k = 3) | 9,00 % ± 4,54 | 11,92 | 14,74 | 12,2 % | 26,8 % | 4 de 5 | 9,00 – 10,22 |
| Ridge (alpha = 1000) | 9,08 % ± 6,04 | 12,42 | 14,23 | 11,5 % | 26,2 % | 2 de 5 | 9,02 – 9,14 |
| Lasso (alpha ≈ 0,032) | 9,40 % ± 5,66 | 12,88 | 15,28 | 8,3 % | 23,6 % | 4 de 5 | 9,30 – 9,47 |
| Predictor de la media | 10,26 % ± 5,86 | 13,94 | 15,82 | — | 16,6 % | 3 de 5 | 10,26 |
| Ingenuo estacional | 12,30 % ± 7,76 | 15,86 | 18,54 | −19,9 % | — | — | 12,30 |
| MLR completa | 14,80 % ± 6,52 | 18,83 | 22,11 | −44,3 % | −20,3 % | 1 de 5 | 14,49 – 14,80 |
| MLR con stepwise | 14,98 % ± 6,73 | 19,04 | 22,31 | −46,0 % | −21,8 % | 1 de 5 | 14,53 – 14,98 |

**Variabilidad.** La desviación entre folds (4,5 a 7,8 puntos) es mucho mayor que la distancia entre
los tres mejores modelos (0,40 puntos): el error depende más del periodo que del modelo. El fold 5
(febrero a julio de 2025) es el más difícil para todos (16,5 % a 21,1 %), porque la serie crece y
ningún modelo lo anticipa; el ingenuo falla sobre todo en el fold 1 (21,3 %), al copiar 2022. Entre
escenarios el MAPE nacional casi no cambia (rango de 0,12 puntos en Ridge y 0,17 en Lasso; cero en
las líneas base, porque la serie nacional es real), salvo en KNN (1,22 puntos), que baja del primer al
tercer puesto en el escenario alto. Con MAD el orden es el mismo que con MAPE; con RMSE Ridge
pasa al primer lugar (14,23 frente a 14,74 de KNN): KNN tiene menos error típico pero algún mes con
un error mayor. En todos los modelos el RMSE es entre 1,13 y 1,24 veces el MAD, así que ningún modelo
depende de unos pocos meses desastrosos.

**Comparación por fold.** Como todos los modelos usan los mismos 5 folds, se restan fold a fold.
KNN y Lasso ganan al ingenuo estacional en 4 de 5 folds; Ridge solo en 2, y su ventaja media
descansa en el fold 1. Frente a la media, la línea base más difícil, la ventaja es más estrecha:
Lasso gana en 4 de 5 folds, KNN y Ridge en 3 (1,25 y 1,18 puntos menos en promedio). Los tres, y
también la media, ganan a la MLR con stepwise en 4 de 5. Con 5
folds no hay potencia para una prueba de hipótesis, por eso se reporta el conteo y no p-valores.

**Modelo preliminar.** Ridge con alpha = 1000 (`docs/decisiones.md`, 27 de octubre). En el
escenario central empata con KNN (9,08 % frente a 9,00 %, una diferencia menor que la desviación
entre folds) y se prefiere por estabilidad: su MAPE varía 0,12 puntos entre escenarios y el de KNN
1,22. Además, sus coeficientes se pueden interpretar (7.4, interpretación).

**Limitaciones.**

- El MAPE de validación del modelo elegido es algo optimista, porque los hiperparámetros y el modelo
  se eligieron con esos mismos folds. La estimación honesta será la evaluación única en la prueba,
  en la entrega final.
- `_rez1` supone que el dato de t−1 ya está publicado al pronosticar t. Según las fechas de
  `data/raw/FUENTES.md`, las importaciones y las licencias de construcción (ELIC) salen unos 45 a 50
  días después del cierre del mes, así que en la práctica llegarían con 2 o 3 meses de rezago.
- El ingenuo estacional usa el valor de t−12 y los modelos los predictores de t−1, lo que favorece
  levemente a los modelos.
- En KNN, las 56 filas de un mes comparten los predictores nacionales y hay empates exactos de
  distancia entre vecinos. `algorithm="kd_tree"` y un solo hilo hacen que el desempate sea siempre el
  mismo, pero la regla de desempate es arbitraria: otra igual de válida cambia el MAPE de KNN en
  centésimas a décimas de punto (`docs/decisiones.md`, 26 de octubre), lo bastante para alterar su
  puesto en el escenario alto, donde queda a 0,03 puntos de la media. Por la misma razón, KNN
  depende del orden de las filas dentro de cada mes, que el ordenamiento por fecha de la carga no
  fija y que puede variar entre computadores: las cifras oficiales son las de
  `results/tablas/cv_resultados.csv`, que reproducen dos computadores distintos.

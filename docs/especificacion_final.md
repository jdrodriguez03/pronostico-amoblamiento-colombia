# Especificación final congelada

> Dueño: Juanes. Entrega final, lunes 9 de noviembre de 2026. Responde a la corrección 5 de la profesora (la prueba se abre una sola vez) y a la 4 (Ridge y KNN como candidatos).

Este documento fija **todo** lo que define el modelo antes de abrir la prueba (agosto 2025 – julio 2026). Lo que está aquí no cambia después de ver la prueba: si la prueba contradice a la validación, se reporta y se discute, no se corrige. El código que lo implementa ya está en el repositorio; cada fila dice dónde.

**Commit de congelamiento:** `________` (se anota el hash del merge de esta rama a `main`, antes de correr `scripts/evaluar_prueba.py`).

## 1. Qué se pronostica

El índice de ventas reales (2019 = 100) del mes t, por departamento × línea (56 series), con el pronóstico emitido el primer día hábil del mes t. Es un índice, no unidades, pesos ni inventario. La cifra principal es la serie **nacional** de la línea 8 (electrodomésticos y muebles), que se obtiene agregando los departamentos.

## 2. Variables y rezagos

Los predictores entran con el rezago con que de verdad están publicados el día de emisión (`REZAGOS_REALES` de `src/features/rezagos.py`, Julián; detalle en `docs/disponibilidad_datos.md`).

| Variable | Columna que entra al modelo | Rezago k | Transformación |
| --- | --- | --- | --- |
| Índice de confianza del consumidor | `icc_rez2` | 2 | estandarizar |
| Inflación anual | `inflacion_anual_rez2` | 2 | estandarizar |
| Tasa de interés de consumo | `tasa_consumo_rez1` | 1 | estandarizar |
| Importaciones de muebles (USD miles) | `importaciones_muebles_usd_miles_rez3` | 3 | estandarizar |
| Área licenciada de vivienda (m²) | `area_vivienda_m2_rez3` | 3 | logaritmo y luego estandarizar |
| Mes, departamento, línea | `mes`, `departamento`, `linea` | se conocen de antemano | one-hot |
| Confinamiento 2020 | `covid` | se conoce de antemano | pasa sin cambios (0/1) |

- El rezago se aplica dentro de cada serie departamento × línea, ordenada por fecha, y solo mira al pasado. Con k máximo de 3, el entrenamiento empieza en **mayo de 2019**: 75 meses × 56 series = 4.200 filas (en la Entrega 2 eran 78 meses).
- En la prueba se concatenan entrenamiento y prueba antes de aplicar el rezago, para que agosto de 2025 tome su dato de mayo a julio de 2025, y luego se vuelven a separar.
- Las columnas de `componentes_regionales.csv` y `lineas_nacionales.csv` nunca son predictores (`config.COLUMNAS_PROHIBIDAS`).
- Código: `preparar_datos(df, "reales")` en `src/modelos/pipeline.py` es el único lugar donde se decide qué columnas entran. Lo usan `entrenar.py` y `evaluar_prueba.py`.

## 3. Preprocesamiento (dentro del Pipeline)

Todo lo que aprende de los datos se ajusta en cada fold, solo con los meses de entrenamiento de ese fold (`construir_pipeline`, `crear_preprocesador`).

- **Objetivo en logaritmo** con `TransformedTargetRegressor` (`LOG_OBJETIVO = True`): el modelo aprende log(índice) y devuelve la predicción en la escala original; todas las métricas se calculan en la escala original.
- **One-hot** de mes, departamento y línea con `handle_unknown="ignore"`:
  - `drop="first"` para los modelos lineales (Ridge, MLR), para que las dummies no sean colineales con el intercepto;
  - **sin** `drop` para KNN, para que todas las categorías queden a la misma distancia (√2) entre sí.
- **`StandardScaler`** para los cinco predictores numéricos (el área de vivienda, después del logaritmo).
- **Orden fijo de filas**: fecha, departamento, línea (ordenamiento estable). Las 56 filas de un mes comparten los predictores nacionales y empatan en distancia en KNN; sin un orden fijo KNN daba entre 8,70 % y 9,36 % según el computador. Ridge no depende del orden.
- Un solo hilo de cálculo (`threadpool_limits(1)`, `n_jobs=1`) y KNN con `algorithm="kd_tree"`, para que los empates se resuelvan igual en cualquier PC.

## 4. Candidatos y rejillas

Solo dos candidatos a modelo final. No se agregan algoritmos. Las rejillas no cambian; el valor lo elige la validación cruzada con la especificación realista.

| Modelo | Papel | Rejilla congelada (`src/modelos/candidatos.py`) |
| --- | --- | --- |
| Ridge | candidato (modelo por defecto) | `alpha` en 17 valores de 10⁻³ a 10⁵ (`np.logspace(-3, 5, 17)`) |
| KNN | candidato (retador) | `n_neighbors` ∈ {1, 2, 3, 5, 10, 20, 40} × `weights` ∈ {uniform, distance} (14 combinaciones) |
| MLR con stepwise | referencia: método del artículo | sin rejilla: α de entrada y salida = 0,25, dummies forzadas |
| Media por serie | línea base baja | sin hiperparámetros |
| Ingenuo estacional (t−12) | línea base exigente | sin hiperparámetros |

Lasso y la MLR completa siguen en la validación cruzada como contraste, pero no son candidatos a modelo final y no se evalúan en la prueba.

## 5. Validación cruzada

- `CVTemporalPorMes`: 5 folds de 6 meses, ventana creciente, partición por **mes** (nunca por fila, nunca `train_test_split` ni `shuffle=True`).
- Con la especificación realista los folds validan los mismos 30 meses que en la Entrega 2, de febrero de 2023 a julio de 2025; el fold 1 entrena con 45 meses (mayo 2019 – enero 2023) y el fold 5 con 69.
- Búsqueda: `GridSearchCV(scoring=scorer_mape_nacional, refit=True, return_train_score=True, n_jobs=1)`, definida una sola vez en `buscar_hiperparametros` (`src/evaluacion/validar.py`).

## 6. Agregación nacional

La serie nacional de la línea 8 es el promedio ponderado de los 7 departamentos con `PESOS_DEPTO` (pesos base 2019), con `agregar_nacional` (`src/evaluacion/agregacion.py`). Se agrega **antes** de calcular el error, tanto lo observado como lo predicho.

## 7. Métricas

- **Principal:** MAPE de la serie nacional de la línea 8 (es la cifra del artículo y la que usa `GridSearchCV` para elegir).
- **Contraste:** MAD y RMSE nacionales (en unidades del índice; RMSE mucho mayor que MAD indica meses con errores grandes). Se reporta también el MSD (el del artículo) y el MAPE del panel (56 series), que no se compara con el nacional.
- **Mejora** frente a no tener modelo: `100 · (MAPE base − MAPE modelo) / MAPE base`, frente a la media y frente al ingenuo estacional.
- **Incertidumbre en prueba:** bootstrap de bloques móviles sobre la serie nacional mensual, bloques de 3 meses consecutivos, B = 2000, semilla 42, IC 95 % por percentiles (`src/evaluacion/incertidumbre.py`). Nunca fila por fila. La diferencia Ridge − KNN se mide con las mismas réplicas (pareada).
- **Rango para la app:** percentiles 10, 50 y 90 del error relativo fuera de muestra (los meses de validación de los 5 folds) del modelo final, por ámbito y línea (`rango_error.csv`).

## 8. Escenario de decisión

El escenario **central** de η (σ = 7 %). Los escenarios bajo y alto se reportan como sensibilidad; la decisión no se toma en ellos.

## 9. Regla para escoger el modelo final

Escrita antes de abrir la prueba. Implementada en `regla_modelo_final` (`src/evaluacion/comparacion.py`) y aplicada por `evaluar_prueba.py`.

> El modelo final es Ridge. Se cambia a KNN solo si se cumplen las dos condiciones en el escenario central: (a) el MAPE nacional de prueba de KNN es menor que el de Ridge por más de la desviación estándar entre folds del MAPE de validación de Ridge, y (b) KNN tiene menor error absoluto que Ridge en al menos 8 de los 12 meses de prueba. Ridge se prefiere a igualdad práctica por estabilidad e interpretabilidad.

Precisiones para que no haya nada que interpretar después:

- La desviación de (a) es la desviación muestral (`ddof = 1`) de los 5 MAPE de validación de Ridge con su alpha ganador, en el escenario central y con la especificación realista. Es la `mape_cv_desv` de Ridge en `prueba.csv` y la `mape_desv` de `comparacion_cv.csv`.
- (a) es una desigualdad estricta: MAPE Ridge − MAPE KNN > desviación.
- En (b), un mes en que los dos errores absolutos son exactamente iguales no cuenta como mes ganado por KNN.
- Ejemplo: Ridge 8,0 %, KNN 7,0 % y desviación 0,9 → (a) se cumple (1,0 > 0,9). Si KNN tiene menor error en 7 de los 12 meses, (b) no se cumple y el modelo final sigue siendo Ridge.

## 10. Cómo se abre la prueba

Una sola vez, el martes 10 entre las 12:00 y las 14:00, solo Juanes, con `python scripts/evaluar_prueba.py --escenario todos`, y solo cuando el commit de congelamiento esté en `main`. Antes de eso nadie calcula una métrica, una media ni una gráfica con esos meses. Si `results/tablas/prueba.csv` ya existe, el script se niega a correr (solo `--forzar` lo permite, para el clon limpio de reproducibilidad).

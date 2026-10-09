# Pronóstico Amoblamiento: guía día por día (Entrega 2)

> Versión en el repo de la guía compartida del equipo. Si hay diferencias, manda la versión del documento compartido.

Oct 8, 2026 · @Julian

En 20 días y 7 bloques llegamos al Oct 28, 2026 con el dataset consolidado, el flujo sin fuga, cinco modelos comparados en tres escenarios, el informe de 3 a 5 páginas y la sustentación ensayada.

## Cómo usar esta guía

Cada persona busca su nombre dentro del día en que va y sigue los pasos en orden, marcando las casillas. Al final de cada día debe poder mostrar lo que dice **Listo cuando**. Si algo no queda claro, comenten sobre el paso exacto. Lean primero la *guía rápida del repositorio*: ahí están las carpetas, el contrato del dataset y las reglas.

**Fechas:** día 1 = viernes 9 de octubre; día 20 = miércoles 28 de octubre, entrega. El lunes 12 es festivo y se trabaja igual, con menos horas. Las horas son estimadas: entre 2 y 4 por día por persona, un poco más los fines de semana.

### Quién es quién

| Persona | Letra | Rol | Responde en el informe por |
| --- | --- | --- | --- |
| Aleja | A | Repositorio, variable objetivo y simulación; auditora de fuga; arma el informe | 7.1 Consolidación de los datos, formato y ortografía |
| Julián | B | Predictores, modelos candidatos e interpretación | 7.3 Entrenamiento y comparación; interpretación de la 7.4; referencias y uso de IA |
| Juanes | C | Validación, Pipeline, métricas y resultados | 7.2 Flujo sin fuga; validación y métricas de la 7.4; cuadre de cifras y guion |

La letra se usa en el nombre de las ramas (`a/e1-repo`, `b/e2-predictores`, `c/e3-pipeline`).

### Rutina de cada día

1. **Al empezar (5 min, por el chat):** escriban qué van a hacer hoy, copiando los pasos de esta guía.
2. **Durante el día:** trabajen solo en su rama y sus archivos. Si necesitan algo de otra persona, usen el fixture; no esperen.
3. **Al terminar (5 min, por el chat):** tres líneas: qué terminé, qué me falta, qué me bloquea. Hagan `git push` aunque no esté terminado.
4. **Día de cierre de bloque (45 min, en llamada, 19:00):** cada quien abre su PR antes de las 18:00; se revisan juntos, se fusionan y Aleja etiqueta el bloque (`e1`, `e2`...). Después, `git checkout main && git pull`.
5. **Cruce de conocimiento (30 min, después de los cierres 2, 3 y 5):** una persona explica su parte a las otras dos, que le hacen las preguntas que haría el profesor.

### Decisiones que se toman el día 1

Cuatro decisiones cambian el trabajo de todos. La guía y `config.py` ya asumen la opción recomendada; si en el arranque se decide otra, se ajustan los pasos afectados. La ventana de pandemia ya quedó fijada en la Entrega 1 (`covid` = abril a agosto de 2020).

| Decisión | Recomendado | Por qué |
| --- | --- | --- |
| ¿Predictores del mismo mes o rezagados? | Rezagados (`_rez1`), `USAR_REZAGO = True`, y se descarta enero de 2019 | Al pronosticar el mes t aún no se conocen sus datos económicos; descartar el único mes sin rezago de vivienda evita imputar |
| ¿Modelar el índice en logaritmo? | Sí, con `TransformedTargetRegressor` | El índice se construyó como producto L × F × η: en logaritmo esos factores se suman |
| ¿Agregar KNN como cuarto modelo? | Sí, KNN para regresión | La rúbrica de la 7.3 nombra KNN y pide "varios modelos" |
| ¿Cómo ve el profesor el código? | Repo privado hasta el 27; ese día se invita al profesor o se hace público | Según lo que pida el curso |

Los modelos que se comparan quedan así: **MLR con stepwise** (reproduce el artículo), **MLR completa** (todas las variables, para ver la multicolinealidad), **Ridge**, **Lasso** y **KNN**, contra dos líneas base: **predictor de la media** e **ingenuo estacional**.

## El mapa de los 20 días

| Bloque | Días | Fechas | Cierre |
| --- | --- | --- | --- |
| 1 · Cimientos | 1–2 | 9–10 oct | e1 |
| 2 · Evidencia de la base (7.1) | 3–5 | 11–13 oct | e2 |
| 3 · Flujo sin fuga (7.2) | 6–8 | 14–16 oct | e3 |
| 4 · Modelos y comparación (7.3) | 9–11 | 17–19 oct (punto de control el 19) | e4 |
| 5 · Prueba e interpretación (7.4) | 12–14 | 20–22 oct (se abre la prueba el 20) | e5-resultados |
| 6 · Informe | 15–17 | 23–25 oct | e6 |
| 7 · Revisión y entrega | 18–20 | 26–28 oct | v2.0 |

Cada cierre fusiona los PR del bloque y deja una etiqueta en main; el punto de control del 19 decide el modelo preliminar antes de abrir la prueba el 20.

## Bloque 1 · días 1 y 2 · Cimientos

**Objetivo del bloque:** los tres con el repositorio corriendo y la base de la Entrega 1 entendida a fondo; la agregación nacional, la primera mirada a los predictores y la validación cruzada por mes, listas.

**Lo que ya está hecho (Julián, 8 de octubre):** el repositorio `pronostico-amoblamiento-colombia` tiene la base completa y verificada. Los 8 insumos están en `data/raw/`. `src/simulation/simular_base.py` y `src/features/predictores.py` reproducen byte a byte las bases de `data/interim/` y las tres `data/processed/base_final_*.csv`. Además, `config.py` y `esquema.validar_base` coinciden con las 19 columnas reales, y `cargar_entrenamiento` y `cargar_prueba` ya funcionan. Hay 20 pruebas en verde. **Ya no hay que construir la base:** el bloque 1 se dedica a entenderla y a armar las piezas que la usan.

### Día 1 · viernes 9 de octubre

**Objetivo del día:** que a las 19:00 los tres tengan el repositorio corriendo y sepan de dónde sale cada columna.

**Aleja · apropiarse de la base (3 h)** (es la dueña de la sección 7.1 y de `src/simulation/`)

- [ ] Aceptar la invitación, clonar y seguir "Primer arranque en 15 minutos" hasta ver `pytest` en verde.
- [ ] Correr `python src/simulation/simular_base.py` y comprobar con `git status` que ningún archivo cambió: esa es la prueba de reproducibilidad.
- [ ] Leer completos `simular_base.py`, `data/README.md` y `results/verificacion.md`, y entender cada una de las 5 pruebas.
- [ ] Escribir en el chat 3 preguntas que el profesor podría hacer sobre la simulación, con su respuesta (por ejemplo: ¿por qué η = exp(e − σ²/2)?, ¿por qué se reescala para conservar la serie nacional?).

* **Listo cuando:** puede explicar en 3 minutos cómo sale `indice_ventas_real` y por qué la serie nacional de muebles no depende de η.

**Julián · fuentes y presentación de la base (3 h)**

- [ ] Invitar a Aleja y a Juanes en **Settings → Collaborators** (si no lo hizo ya).
- [x] Completar en `data/raw/FUENTES.md` la URL y la fecha de descarga de cada insumo (hecho el 9 de octubre; faltan los enlaces directos de `interes.xlsx` y del PDF de Fedesarrollo, marcados VERIFICAR).
- [ ] Correr `python src/features/predictores.py` (requiere `pdftotext`) y comprobar con `git status` que nada cambió.
- [ ] Preparar 15 minutos para la reunión: las 19 columnas, por qué existen las `_rez1`, por qué el ICC es solo nacional, el capítulo 94 completo y la ventana de `covid`.

* **Listo cuando:** `FUENTES.md` está completo y la explicación está lista.

**Juanes · diseñar la validación (3 h)**

- [ ] Releer la sección 6.4 de la Entrega 1 y los criterios 7.2 y 7.4 de la rúbrica.
- [ ] Revisar la tabla de folds de `docs/validacion.md`: con `USAR_REZAGO = True` el entrenamiento va de febrero de 2019 a julio de 2025 (78 meses) y el fold 1 entrena con 48 meses.
- [ ] Escribir la sección "Por qué así": ventana creciente; partición por mes, porque las 56 filas de un mes comparten los predictores nacionales; y el ingenuo estacional siempre tiene su t−12 en entrenamiento.
- [ ] En `notebooks/03_comparacion_modelos.ipynb`, cargar `cargar_entrenamiento("central")` y graficar la serie nacional de muebles, para conocer los datos.

* **Listo cuando:** `docs/validacion.md` está completo.

**Todos · reunión de arranque (1.5 h, 19:00, en llamada)**

- [ ] (10 min) Leer juntos "La idea en tres líneas" y "Las 6 reglas" de la guía del repositorio.
- [ ] (25 min) Julián recorre el repositorio y la base: carpetas, scripts de construcción, columnas y archivos de auditoría.
- [ ] (15 min) Revisar el contrato (`COLUMNAS_BASE` en `esquema.py`). Después de hoy solo cambia en un cierre.
- [ ] (20 min) Tomar las decisiones del día 1 y que Juanes las escriba en `docs/decisiones.md`.
- [ ] (20 min) Cada quien sigue el primer arranque y corre `pytest`. Nadie cuelga hasta que los tres vean verde.

### Día 2 · sábado 10 de octubre

**Objetivo del día:** primeras piezas probadas y fusionadas en main.

**Aleja · agregación nacional (3.5 h) · rama `a/e1-agregacion`**

- [ ] En `src/evaluacion/agregacion.py`, implementar `agregar_nacional(df, columna, linea_codigo=8)`: filtra la línea, multiplica por `PESOS_DEPTO` de cada departamento y suma por fecha. Devuelve una serie indexada por fecha.
- [ ] `tests/test_agregacion.py`: `agregar_nacional(base, "indice_ventas_real")` coincide con `L_8` de `lineas_nacionales.csv` (error relativo < 1e-6) en los tres escenarios, y lo mismo con la línea 4 y `L_4`.

* **Listo cuando:** la prueba pasa. Esta función es la que usa Juanes para el MAPE nacional.

**Julián · primera mirada a los predictores (4 h) · rama `b/e1-predictores`**

- [ ] En `src/evaluacion/interpretacion.py`, `calcular_vif(df, columnas)` con `variance_inflation_factor` de statsmodels (agregando la constante).
- [ ] En `notebooks/02_calidad_predictores.ipynb`, **solo con `cargar_entrenamiento`**: tabla descriptiva de los 10 predictores, matriz de correlación y cada serie en el tiempo con los meses de `covid` sombreados.
- [ ] VIF de los 5 predictores rezagados en `results/tablas/vif_inicial.csv`.
- [ ] Proponer transformaciones en `docs/diccionario_datos.md` (por ejemplo, logaritmo de importaciones y de área de vivienda), con su porqué.

* **Listo cuando:** la tabla de VIF y la figura de series están en la rama.

**Juanes · fixture y particiones (4 h) · rama `c/e1-particion`**

- [ ] `scripts/crear_fixture.py`: `data/fixtures/base_mini.csv`, un recorte de `base_final_central.csv` con Antioquia y Bogota, líneas 4 y 8, y los 91 meses (364 filas). Debe pasar `validar_base`.
- [ ] En `src/evaluacion/particion.py`, `CVTemporalPorMes` según la tabla de `docs/validacion.md`, leyendo la columna `fecha`.
- [ ] `tests/test_particion.py`: ningún mes está en entrenamiento y validación del mismo fold, la validación siempre es posterior, las ventanas coinciden con la tabla y ningún mes desde agosto de 2025 aparece.

* **Listo cuando:** la prueba pasa con `cargar_entrenamiento("central")`.

**Cierre del bloque 1 (todos, 19:00, 45 min):** se fusionan los tres PR y Aleja etiqueta `e1`.

## Bloque 2 · días 3 a 5 · Evidencia de la base (7.1) y piezas del flujo

**Objetivo del bloque:** la evidencia visual de que la simulación reproduce la estructura prevista (sección 7.1), y listos el stepwise, los candidatos, las métricas, las líneas base y el Pipeline. Todos parten de `e1`.

### Día 3 · domingo 11 de octubre

**Objetivo del día:** cada pieza funciona en su caso normal.

**Aleja · verificación visual I (4 h) · rama `a/e2-verificacion`**

`results/verificacion.md` ya tiene las cifras; el informe necesita verlas. En `notebooks/01_verificacion_simulacion.ipynb`:

- [ ] Recuperar el componente simulado de cada escenario: log η = log(índice / (L × F)), con `lineas_nacionales.csv` y `componentes_regionales.csv`. Aquí sí se usan, porque es auditoría, no modelamiento.
- [ ] **Magnitud:** histograma de log η por escenario con la desviación objetivo (2 %, 7 %, 13 %) marcada.
- [ ] **Persistencia:** diagrama de caja de la autocorrelación de orden 1 de las 56 series, con 0,6 marcado.
- [ ] Guardar las figuras con `guardar_figura()`.

* **Listo cuando:** las dos figuras están y coinciden con las cifras de `verificacion.md`.

**Julián · stepwise dentro del Pipeline (4 h) · rama `b/e2-stepwise`**

- [ ] En `src/modelos/stepwise.py`, `StepwiseSelector(BaseEstimator, SelectorMixin)` con `alfa_entrada`, `alfa_salida` (0,25 por defecto) y `forzadas` (índices de las dummies de mes, departamento y línea, que siempre quedan).
- [ ] `fit(X, y)`: stepwise bidireccional como Minitab, con p-valores de OLS de statsmodels. `_get_support_mask()` devuelve las elegidas.
- [ ] Anotar en `docs/decisiones.md` la limitación: las 56 filas de un mes comparten los predictores nacionales, así que los p-valores son optimistas. Por eso se compara con Lasso.
- [ ] `tests/test_stepwise.py`: con la base pequeña más 3 columnas de ruido, el ruido no entra y las forzadas siempre quedan.

* **Listo cuando:** la prueba pasa.

**Juanes · métricas y líneas base (4 h) · rama `c/e2-metricas`**

- [ ] En `src/evaluacion/metricas.py`: `mape`, `mad`, `msd` y `mejora_pct`.
- [ ] En `src/modelos/lineas_base.py`: `PredictorMedia` (promedio de entrenamiento de cada serie departamento × línea) e `IngenuoEstacional` (el valor de la misma serie 12 meses antes).
- [ ] Pruebas: un ejemplo de 3 valores calculado a mano por métrica, MAPE 0 con predicción perfecta, y que el ingenuo devuelve exactamente y(t−12) en la base pequeña.

* **Listo cuando:** las pruebas pasan.

### Día 4 · lunes 12 de octubre (festivo)

**Objetivo del día:** la tabla de verificación y los candidatos definidos.

**Aleja · verificación visual II (3.5 h)**

- [ ] **Conservación:** la serie nacional de muebles reconstruida con `agregar_nacional` y `L_8` en la misma gráfica, con la diferencia máxima.
- [ ] **Fidelidad regional:** correlación por departamento entre la línea de muebles y `indice_grupo_depto`, en los tres escenarios.
- [ ] **Patrones y proporciones:** abril de 2020 frente a la serie real, y participación de cada departamento frente a `PESOS_DEPTO`.
- [ ] `results/tablas/verificacion_simulacion.csv`: prueba, objetivo y obtenido en cada escenario, y si cumple.

* **Listo cuando:** la tabla tiene las 5 pruebas × 3 escenarios y una figura de 2 o 3 paneles para el informe.

**Julián · modelos candidatos (3.5 h)**

- [ ] En `src/modelos/candidatos.py`, `CANDIDATOS` con nombre → (modelo, selector, rejilla):

| Nombre | Modelo | Rejilla de hiperparámetros |
| --- | --- | --- |
| `mlr_stepwise` | `LinearRegression` + `StepwiseSelector` | Ninguna: α = 0,25 como el artículo |
| `mlr_completa` | `LinearRegression` con todas las variables | Ninguna |
| `ridge` | `Ridge` | alpha en `np.logspace(-3, 3, 13)` |
| `lasso` | `Lasso(max_iter=20000)` | alpha en `np.logspace(-4, 1, 11)` |
| `knn` | `KNeighborsRegressor` | n\_neighbors en \[3, 5, 10, 20, 40\]; weights en \["uniform", "distance"\] |

- [ ] Prueba: los 5 se construyen y entrenan sobre la base pequeña (con un Pipeline mínimo mientras llega el de Juanes).

* **Listo cuando:** los 5 entrenan sin errores.

**Juanes · preprocesamiento y Pipeline (3.5 h)**

- [ ] `crear_preprocesador()`: `OneHotEncoder(drop="first", handle_unknown="ignore")` para `mes`, `departamento` y `linea`; `StandardScaler` para `predictores_modelo()`; `covid` pasa sin cambios; y las transformaciones que propuso Julián (por ejemplo logaritmo) con `FunctionTransformer`, **dentro** del preprocesador.
- [ ] `construir_pipeline(modelo, selector=None, log_objetivo=True)`: preprocesador, selector y modelo, envuelto en `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)`.
- [ ] Prueba de fuga: la media del `StandardScaler` ajustado en el fold 1 es la de esos meses, no la de toda la base.

* **Listo cuando:** el Pipeline entrena en la base pequeña y la prueba de fuga pasa.

### Día 5 · martes 13 de octubre

**Objetivo del día:** auditoría de fuga y evaluación cruzada listas; borrador de la 7.1.

**Aleja · auditoría de fuga y borrador 7.1 (4 h)**

- [ ] `tests/test_fuga.py`: `validar_base` falla con cualquier columna de auditoría; `cargar_entrenamiento` nunca devuelve meses desde agosto de 2025; ningún archivo salvo `scripts/evaluar_prueba.py` llama a `cargar_prueba` (buscarlo en el texto); `CVTemporalPorMes` no mezcla meses.
- [ ] `docs/informe/borrador_7_1.md` (máximo una página): la base definitiva, cómo se generó la simulación, la tabla de verificación con una figura, estado de calidad y transformaciones.

* **Listo cuando:** `test_fuga.py` pasa y el borrador cita la tabla y la figura.

**Julián · diccionario y calidad (3 h)**

- [ ] Completar en `docs/diccionario_datos.md` las transformaciones acordadas con Juanes y su porqué.
- [ ] Estado de calidad para el informe: meses atípicos (2020), volatilidad de ELIC, ICC solo nacional, capítulo 94 completo. Una línea cada uno en `docs/decisiones.md` si no estaba.

* **Listo cuando:** el diccionario y las decisiones están al día.

**Juanes · validación cruzada lista para usar (4 h)**

- [ ] En `src/evaluacion/validar.py`, `scorer_mape_nacional(estimador, X, y)`: predice, agrega la línea 8 con `agregar_nacional` y devuelve −MAPE (scikit-learn maximiza).
- [ ] `evaluar_cv(pipeline, df, cv)`: MAPE, MAD y MSD nacionales y MAPE del panel en cada fold.
- [ ] Probarlo con `LinearRegression` y las dos líneas base sobre la base pequeña.

* **Listo cuando:** `evaluar_cv` devuelve una tabla de 5 folds.

**Cierre del bloque 2 (todos, 19:00, 45 min) + cruce de conocimiento 1 (30 min):** se fusionan los PR y se etiqueta `e2`. Después, **Aleja explica la base y su verificación**; Julián y Juanes deben poder responder: ¿qué parte del dato es real y cuál simulada?, ¿por qué la evaluación principal no depende de η?

## Bloque 3 · días 6 a 8 · Flujo sin fuga y primera corrida (7.2)

**Objetivo del bloque:** que cualquier modelo se entrene con un solo comando dentro del Pipeline, con las líneas base en datos reales y una primera corrida completa del escenario central. Gracias a la base ya hecha, este bloque llega con un día de holgura. Todos parten de `e2`.

### Día 6 · miércoles 14 de octubre

**Objetivo del día:** el script de entrenamiento y la reproducibilidad de punta a punta.

**Juanes · script de entrenamiento (4 h) · rama `c/e3-entrenar`**

- [ ] `scripts/entrenar.py --escenario central|bajo|alto|todos [--rapido]`: para cada candidato, `GridSearchCV(pipeline, rejilla, cv=CVTemporalPorMes(), scoring=scorer_mape_nacional, refit=True)` sobre `cargar_entrenamiento(escenario)`. Las líneas base entran por `evaluar_cv`.
- [ ] Guardar `results/tablas/cv_resultados.csv`: escenario, modelo, hiperparámetros elegidos, fold, mape, mad, msd. Los modelos ajustados van a `results/modelos/` (en `.gitignore`).
- [ ] `--rapido` usa la base pequeña y rejillas de 2 valores, para el CI.

* **Listo cuando:** `python scripts/entrenar.py --rapido` corre en menos de 1 minuto.

**Julián · herramientas de interpretación (3.5 h) · rama `b/e3-interpretacion`**

- [ ] En `src/evaluacion/interpretacion.py`: `coeficientes(pipeline)`, que devuelve los coeficientes estandarizados con el nombre de cada variable, y `estabilidad(resultados_por_fold)`, con la media ± desviación de cada coeficiente entre folds.
- [ ] Pruebas con la base pequeña.

* **Listo cuando:** las dos funciones devuelven tablas con nombres legibles.

**Aleja · reproducibilidad de punta a punta (3 h) · rama `a/e3-reproducir`**

- [ ] `scripts/reproducir.py --hasta datos|cv|prueba`: corre en orden `simular_base.py`, `predictores.py`, `entrenar.py` y `evaluar_prueba.py`, según el argumento.
- [ ] Al final imprime el hash MD5 de cada archivo de `data/interim/` y `data/processed/`. Dos corridas seguidas deben dar los mismos hashes.
- [ ] Actualizar la sección "Cómo reproducir" del README.

* **Listo cuando:** `python scripts/reproducir.py --hasta datos` deja `git status` limpio.

### Día 7 · jueves 15 de octubre

**Objetivo del día:** líneas base reales y la primera corrida de todos los candidatos.

**Juanes · líneas base con datos reales (3.5 h)**

- [ ] Correr `evaluar_cv` con `PredictorMedia` e `IngenuoEstacional` en los tres escenarios.
- [ ] Guardar `results/tablas/cv_lineas_base.csv` (escenario, modelo, fold, mape, mad, msd) y una figura de la serie nacional de muebles con el ingenuo estacional en las 5 ventanas de validación.
- [ ] Anotar en el chat el MAPE medio de cada base: es la vara que deben superar los modelos.

* **Listo cuando:** la tabla tiene 3 escenarios × 2 bases × 5 folds = 30 filas.

**Julián · primera corrida real (3 h)**

- [ ] Traer la rama de Juanes sin editarla (`git fetch` y `git checkout c/e3-entrenar`) y correr `python scripts/entrenar.py --escenario central`.
- [ ] Revisar avisos de convergencia de Lasso; si aparecen, ajustar la rejilla en `candidatos.py` (su archivo).
- [ ] ¿Algún modelo supera al ingenuo estacional? ¿Qué variables elige el stepwise? Anotar en `docs/bitacora_resultados.md`.

* **Listo cuando:** hay una primera tabla de los 5 modelos en el escenario central, comentada en la bitácora.

**Aleja · borrador 7.1 v2 (3 h)**

- [ ] Pulir `docs/informe/borrador_7_1.md` con la figura final de verificación y una frase de lectura por figura y tabla.
- [ ] Revisar los PR abiertos de Julián y Juanes buscando un `.fit` fuera del Pipeline o una lectura de la prueba; comentarlo en el PR si lo hay.

* **Listo cuando:** el borrador está completo y los PR revisados.

### Día 8 · viernes 16 de octubre

**Objetivo del día:** borrador de la 7.2 y todo listo para correr los tres escenarios.

**Juanes · figura y texto del flujo (3 h)**

- [ ] Figura `results/figuras/flujo_pipeline.png`: partición por mes → validación cruzada temporal → \[transformar, codificar y escalar → stepwise o penalización → modelo\] ajustado en cada fold → agregación nacional → métricas.
- [ ] Borrador `docs/informe/borrador_7_2.md` (máximo media página): el orden de las etapas y, para cada una, qué información del futuro se filtraría si estuviera fuera del Pipeline. Incluir por qué se usan los predictores rezagados.

* **Listo cuando:** el borrador explica las 6 reglas de oro con ejemplos del proyecto.

**Julián · adelantar la 7.3 (3 h)**

- [ ] Esqueleto de `docs/informe/borrador_7_3.md`: modelos comparados, cómo se ajustan los hiperparámetros, y tabla vacía "artículo frente a este trabajo" (MAPE 4,89 %, variables, signos, VIF máximo).

* **Listo cuando:** el esqueleto está en la rama y solo falta llenarlo con resultados.

**Aleja · preparar el análisis de escenarios (2.5 h)**

- [ ] En `notebooks/05_escenarios.ipynb`, dejar listo el código que leerá `cv_resultados.csv` de los tres escenarios y comparará el orden de los modelos (Spearman), para correrlo el día 10.

* **Listo cuando:** el notebook corre con la tabla del escenario central.

**Cierre del bloque 3 (todos, 19:00, 45 min) + cruce de conocimiento 2 (30 min):** se fusionan los PR y se etiqueta `e3`. Después, **Juanes explica el Pipeline y la validación**; Aleja y Julián deben poder responder: ¿por qué no `train_test_split`?, ¿por qué el stepwise va dentro del Pipeline?, ¿qué pasaría si escaláramos antes de partir?, ¿por qué los predictores rezagados?

## Bloque 4 · días 9 a 11 · Entrenamiento y comparación (7.3)

**Objetivo del bloque:** los 5 modelos y las 2 líneas base comparados en los 3 escenarios con validación cruzada, los hiperparámetros justificados con curvas de validación y el modelo preliminar elegido **sin mirar la prueba**. Es el bloque más importante: si el punto de control del día 11 falla, el bloque 5 se usa para cerrarlo. Todos parten de `e3`.

### Día 9 · sábado 17 de octubre

**Objetivo del día:** todas las corridas hechas y las curvas de validación dibujadas.

**Juanes · corrida en los tres escenarios (3 h) · rama `c/e4-comparacion`**

- [ ] `python scripts/entrenar.py --escenario todos`.
- [ ] `results/tablas/comparacion_cv.csv`: una fila por escenario y modelo (3 × 7 = 21 filas) con MAPE medio ± desviación entre folds, MAD, MSD, mejora % frente a la media y frente al ingenuo estacional, e hiperparámetros elegidos.
- [ ] Compartir en el chat la tabla del escenario central.

* **Listo cuando:** la tabla tiene las 21 filas.

**Julián · curvas de validación (4 h) · rama `b/e4-curvas`**

- [ ] Ridge y Lasso: MAPE nacional de validación contra alpha (escala log), con banda de ±1 desviación entre folds y el alpha elegido marcado.
- [ ] KNN: MAPE contra k, una línea por tipo de peso.
- [ ] Trayectoria de Lasso: coeficiente de cada predictor contra alpha. ¿Cuáles se van a cero primero? Comparar con lo que elige el stepwise.
- [ ] Guardar las 3 figuras en `results/figuras/` con `guardar_figura()`.

* **Listo cuando:** las 3 figuras muestran el hiperparámetro elegido y la variabilidad.

**Aleja · estabilidad de la selección (3 h) · rama `a/e4-escenarios`**

- [ ] Para cada fold y escenario (15 ajustes), registrar qué variables elige el stepwise y cuáles quedan distintas de cero en Lasso.
- [ ] `results/tablas/estabilidad_seleccion.csv`: variable × (veces elegida por stepwise de 15, veces distinta de cero en Lasso de 15).
- [ ] Una frase: ¿la selección es estable o cambia según el periodo?

* **Listo cuando:** la tabla cubre los 5 predictores rezagados y covid.

### Día 10 · domingo 18 de octubre

**Objetivo del día:** la comparación se ve en una figura y se contrasta con el artículo.

**Juanes · figura de comparación (3 h)**

- [ ] En `notebooks/03_comparacion_modelos.ipynb`: MAPE de cada fold por modelo (un punto por fold y la media marcada), ordenado de mejor a peor, con el ingenuo estacional como línea de referencia. Un panel por escenario.
- [ ] Revisar que la figura se lea reducida a media página.

* **Listo cuando:** la figura muestra de un vistazo qué modelos superan al ingenuo estacional.

**Julián · contraste con el artículo (3 h)**

- [ ] Tabla "artículo frente a este trabajo": MAPE (4,89 % en el artículo), variables retenidas (las 5 del artículo y las nuestras), signo de cada coeficiente frente al esperado, VIF máximo.
- [ ] Responder en `docs/bitacora_resultados.md` la pregunta de la E1: ¿la regularización mejora la capacidad predictiva frente a la MLR cuando hay predictores correlacionados?

* **Listo cuando:** la tabla está llena y la pregunta tiene una respuesta preliminar con cifra.

**Aleja · sensibilidad a los escenarios (3 h)**

- [ ] En `notebooks/05_escenarios.ipynb`: ¿cambia el orden de los modelos entre σ = 2 %, 7 % y 13 %? Correlación de Spearman de los rankings y una figura.
- [ ] Explicar por qué el MAPE nacional casi no debería variar (la conservación hace que la serie nacional sea real) mientras el MAPE del panel sí.

* **Listo cuando:** hay una conclusión de una frase: "las conclusiones (no) dependen del componente simulado porque...".

### Día 11 · lunes 19 de octubre · punto de control

**Objetivo del día:** decidir con evidencia cuál es el mejor modelo, antes de mirar la prueba.

**Cada uno (2 h):** cerrar lo que le falte del bloque y abrir su PR antes de las 18:00. Julián escribe además el borrador `docs/informe/borrador_7_3.md` (máximo una página): qué se comparó, cómo se ajustaron los hiperparámetros y qué dice la evidencia frente a las líneas base.

**Cierre del bloque 4 y punto de control (todos, 19:00, 1.5 h):** se fusionan los PR, se etiqueta `e4` y se recorre esta lista en main:

- [ ] `python scripts/reproducir.py --hasta cv` regenera todo desde `data/raw/` sin errores.
- [ ] Hay 5 modelos y 2 líneas base × 3 escenarios × 5 folds en `cv_resultados.csv`.
- [ ] Ridge, Lasso y KNN tienen su hiperparámetro elegido por validación cruzada y su curva.
- [ ] `pytest` pasa, incluido `test_fuga.py`.
- [ ] Nadie ha corrido `evaluar_prueba.py` ni calculado nada con agosto de 2025 a julio de 2026.
- [ ] Se elige el **modelo preliminar** por MAPE medio de validación en el escenario central, desempatando por estabilidad (menor desviación entre folds) e interpretabilidad. Se escribe en `docs/decisiones.md` con sus cifras.

Si todo está marcado, el bloque 5 sigue como está. Si falta algo, cada casilla vacía se asigna a su dueño y se cierra el martes 20 antes de correr la prueba.

## Bloque 5 · días 12 a 14 · Prueba, variabilidad e interpretación (7.4)

**Objetivo del bloque:** la evaluación única en prueba, los resultados con su variabilidad, la primera lectura del mejor modelo y los resultados congelados. Desde el día 12 no se cambia ningún hiperparámetro ni modelo. Todos parten de `e4`.

### Día 12 · martes 20 de octubre

**Objetivo del día:** abrir la prueba una sola vez y empezar a interpretar.

**Juanes · evaluación única en prueba (3 h) · rama `c/e5-prueba`**

- [ ] `scripts/evaluar_prueba.py`: para cada escenario y modelo, reentrenar con la configuración elegida en la validación cruzada sobre febrero de 2019 a julio de 2025 (enero de 2019 se descarta por el rezago) y predecir agosto de 2025 a julio de 2026.
- [ ] Se reportan todos los modelos, pero el modelo preliminar ya se eligió el día 11: lo que salga aquí no cambia la elección. Si el resultado en prueba contradice la validación, se reporta y se discute, no se corrige.
- [ ] `results/tablas/prueba.csv`: escenario, modelo, MAPE, MAD y MSD nacionales, MAPE del panel y mejora % frente a cada línea base.
- [ ] Figura: serie nacional observada en la prueba frente a las predicciones del modelo preliminar, la MLR con stepwise y el ingenuo estacional.
- [ ] Commit con el mensaje `resultados: evaluación única en prueba` y las cifras del escenario central en el chat.

* **Listo cuando:** `prueba.csv` y la figura están en la rama.

**Julián · interpretación del mejor modelo (4 h) · rama `b/e5-interpretacion`**

En `notebooks/04_interpretacion.ipynb`:

- [ ] Coeficientes estandarizados del modelo preliminar, con su signo y magnitud: ¿qué variables pesan más?, ¿tienen el signo que da la teoría económica?
- [ ] Estabilidad: media ± desviación de cada coeficiente entre los 5 folds para MLR completa, Ridge y Lasso. Es la respuesta directa a "¿la regularización mejora la estabilidad?".
- [ ] VIF de las variables elegidas, calculado solo con entrenamiento, junto a los del artículo (16,16 y 23,18).
- [ ] Guardar `results/tablas/coeficientes.csv` y una figura de coeficientes con barras de error.

* **Listo cuando:** la tabla de coeficientes con media ± desviación existe para los 3 modelos lineales.

**Aleja · análisis de errores (3 h) · rama `a/e5-errores`**

- [ ] Residuos de la línea nacional en el tiempo (validación y prueba): ¿en qué meses se equivoca más el modelo?, ¿hay un patrón estacional que no capturó?
- [ ] MAPE del panel por línea y por departamento en un mapa de calor: ¿qué combinaciones son más difíciles?
- [ ] Meses de confinamiento de 2020 (con los residuos de ajuste del modelo entrenado hasta julio de 2025, porque esos meses no caen en ninguna validación): comparar MAPE y MAD, como se anunció en la E1, porque el MAPE se dispara cuando el valor observado es muy bajo.
- [ ] Tres hallazgos en una frase cada uno en `docs/bitacora_resultados.md`.

* **Listo cuando:** hay 2 figuras y 3 hallazgos escritos.

### Día 13 · miércoles 21 de octubre

**Objetivo del día:** la variabilidad cuantificada y la lista de figuras del informe cerrada.

**Juanes · variabilidad (3 h)**

- [ ] Tabla final `results/tablas/tabla_final.csv`: por modelo, MAPE de validación (media ± desviación entre folds), rango entre los 3 escenarios y MAPE de prueba con intervalo de confianza del 95 %.
- [ ] El intervalo sale de un bootstrap sobre los 12 meses de prueba: remuestrear meses con reemplazo 1.000 veces (con `SEMILLA`) y tomar los percentiles 2,5 y 97,5.

* **Listo cuando:** cada modelo tiene sus tres medidas de variabilidad.

**Julián · borrador de la interpretación (3 h)**

- [ ] `docs/informe/borrador_7_4_interpretacion.md` (máximo media página): variables relevantes, tipos de error y sus implicaciones para la planeación de inventarios, usando los hallazgos de Aleja.
- [ ] Pasar la respuesta sobre regularización y estabilidad a una frase con cifra.

* **Listo cuando:** el borrador menciona cada figura y tabla que usa.

**Aleja · lista final de figuras y tablas (3 h)**

- [ ] Elegir con los otros dos, por chat, máximo 6 figuras y 4 tablas para el informe (las demás quedan en el repositorio).
- [ ] Que todas tengan el mismo estilo (`src/figuras.py`), título, ejes con unidades y un nombre de archivo con su número (`fig03_comparacion_cv.png`).
- [ ] `docs/informe/lista_figuras_tablas.md`: número, título, sección, archivo y la frase de lectura que la acompaña.

* **Listo cuando:** la lista existe y cada figura tiene su frase de lectura.

### Día 14 · jueves 22 de octubre

**Objetivo del día:** resultados congelados y reproducidos desde un clon limpio.

**Juanes · borrador de validación y métricas (3 h)**

- [ ] `docs/informe/borrador_7_4_metricas.md` (máximo media página): estrategia de validación, por qué MAPE con MAD de contraste, resultados con su variabilidad (tabla final) y mejora frente a las líneas base.

* **Listo cuando:** cada cifra del borrador está en un CSV de `results/tablas/`.

**Julián · cerrar la 7.3 (2 h)**

- [ ] Actualizar `borrador_7_3.md` con las figuras definitivas y la tabla "artículo frente a este trabajo".

* **Listo cuando:** la 7.3 cabe en una página con su tabla y su figura.

**Aleja · congelar y reproducir desde cero (3 h)**

- [ ] Después de fusionar, en una carpeta nueva: `git clone`, entorno nuevo, `pip install -r requirements.txt && pip install -e .`, `python scripts/reproducir.py --hasta prueba`.
- [ ] Comparar las tablas regeneradas con las de main: deben ser idénticas. Si algo cambia, es un problema de semilla: avisar al dueño.
- [ ] Etiquetar `e5-resultados`. Desde aquí, las cifras del informe salen de esta etiqueta.

* **Listo cuando:** el clon limpio reproduce las mismas tablas.

**Cierre del bloque 5 (todos, 19:00, 45 min) + cruce de conocimiento 3 (30 min):** se fusionan los PR y se etiqueta `e5-resultados`. Después, **Julián explica los modelos y la interpretación**; Aleja y Juanes deben poder responder: ¿por qué ganó ese modelo?, ¿qué variable pesa más y qué significa para una empresa de muebles?, ¿qué hace Lasso que no hace Ridge?

## Bloque 6 · días 15 a 17 · Informe

**Objetivo del bloque:** el informe completo en Word, de 3 a 5 páginas de cuerpo, con cada cifra rastreada a su tabla, y el README que deja reproducir todo. Desde aquí no se escribe código nuevo, solo correcciones. Todos parten de `e5-resultados`.

**Las reglas del informe** (vienen de la rúbrica): portada con título, integrantes y curso; secciones 7.1 a 7.4; figuras y tablas numeradas e interpretadas en el texto; justificar cada decisión sin repetir teoría de clase; código por enlace al repositorio; uso de IA declarado; cero errores de ortografía. Menos de 3 páginas o más de 5 se penaliza.

**Presupuesto de páginas:** 7.1 ≈ 1 página · 7.2 ≈ 0,75 · 7.3 ≈ 1 · 7.4 ≈ 1,25 · introducción breve y conclusiones ≈ 0,5. Total ≈ 4,5, con margen.

### Día 15 · viernes 23 de octubre

**Objetivo del día:** la versión 1 completa.

**Aleja · armar el informe (4 h)**

- [ ] Crear el Word con el mismo formato de la Entrega 1: portada, introducción de 3 líneas (qué se hizo desde la E1), secciones 7.1 a 7.4 pegadas desde los borradores, conclusiones preliminares, referencias, enlace al repositorio y a la etiqueta `e5-resultados`.
- [ ] Insertar las figuras y tablas de `lista_figuras_tablas.md` con su número y pie de fuente.
- [ ] Contar las páginas del cuerpo (sin portada, referencias, figuras ni tablas) y escribirlo en el chat.

* **Listo cuando:** la versión 1 está en la carpeta compartida con el número de páginas.

**Julián · referencias y declaración de IA (2.5 h)**

- [ ] Referencias en el mismo estilo de la E1: artículo base, DANE (EMC, ELIC, IPC, importaciones), Fedesarrollo, Superintendencia Financiera, y las librerías usadas (scikit-learn, statsmodels, pandas), porque la rúbrica exige citar código y librerías externas.
- [ ] `docs/uso_ia.md` y un párrafo en el informe: para qué se usó IA generativa (por ejemplo, organizar el plan de trabajo y revisar código) y que cada decisión técnica fue entendida y puede sustentarse.

* **Listo cuando:** las referencias están completas y la declaración está en el informe.

**Juanes · cuadre de cifras (3 h)**

- [ ] `docs/cuadre_cifras.md`: cada número del informe (MAPE, VIF, número de registros, hiperparámetros, porcentajes) con la sección donde aparece, el archivo CSV y la fila de donde sale.
- [ ] Marcar en el Word con comentario cualquier cifra que no cuadre o que no tenga fuente.

* **Listo cuando:** el 100 % de las cifras está rastreado.

### Día 16 · sábado 24 de octubre

**Objetivo del día:** que alguien que no escribió cada sección la entienda.

**Todos · lectura cruzada (2 h cada uno).** Cada quien lee las secciones que no escribió y deja comentarios en el Word. Aleja lee la 7.2 y la 7.3; Julián, la 7.1 y la 7.2; Juanes, la 7.1 y la 7.3; los tres leen la 7.4. Preguntas que guían la lectura:

- [ ] ¿Cada decisión tiene su porqué en una o dos frases?
- [ ] ¿Cada figura y tabla tiene una frase que dice qué se ve en ella?
- [ ] ¿Hay descripción mecánica ("luego se hizo...") o teoría repetida que se pueda cortar?
- [ ] ¿Entiendo esta sección lo suficiente como para defenderla ante el profesor?

**Aleja · README final (2 h)**

- [ ] README con: qué es el proyecto, artículo base, estructura de carpetas, instalación paso a paso, `python scripts/reproducir.py --hasta prueba`, cuánto tarda, dónde quedan las tablas y figuras, integrantes y declaración de IA.

* **Listo cuando:** el README está en main.

**Juanes · probar el README como si fuera el profesor (1 h)**

- [ ] Seguir el README al pie de la letra en una carpeta nueva, sin usar nada que no diga. Anotar cada paso donde se trabó.

* **Listo cuando:** el README funciona sin ayuda o los fallos están reportados a Aleja.

### Día 17 · domingo 25 de octubre

**Objetivo del día:** la versión 2 del informe, dentro del límite de páginas.

- [ ] **Cada dueño (2 h):** atender los comentarios de su sección directamente en el Word.
- [ ] **Aleja (2 h):** si el cuerpo pasa de 5 páginas, recortar primero repeticiones y teoría, luego mover detalle a una tabla; versión 2.
- [ ] **Juanes (1 h):** repasar el cuadre de cifras sobre la versión 2.
- [ ] **Julián (1 h):** revisar ortografía de 7.3 y 7.4 con el corrector del Word y una segunda lectura en voz alta.

**Cierre del bloque 6 (todos, 19:00, 45 min):** se revisa la versión 2 juntos, se fusionan las correcciones al repositorio y se etiqueta `e6`.

## Bloque 7 · días 18 a 20 · Revisión, sustentación y entrega

**Objetivo del bloque:** informe final en PDF, repositorio accesible para el profesor con la etiqueta `v2.0`, y los tres capaces de sustentar cualquier parte. La guía asume que la sustentación es el 28 o después; si es antes, los ensayos se adelantan. Todos parten de `e6`.

### Día 18 · lunes 26 de octubre

**Objetivo del día:** guion y banco de preguntas listos, primer ensayo.

**Juanes · guion de la sustentación (3 h)**

- [ ] `docs/guion_sustentacion.md`: tabla minuto a minuto con qué se muestra (figura o tabla), qué cifra debe decirse y quién habla. Duración según lo que indique el profesor.
- [ ] Cada persona presenta la parte que **no** construyó (Aleja la 7.3, Julián la 7.2, Juanes la 7.1) y la 7.4 se reparte. Así el profesor ve dominio de todo el equipo.
- [ ] Preparar las diapositivas o la vista del informe que se proyectará.

* **Listo cuando:** el guion tiene nombre en cada renglón.

**Julián · banco de preguntas (3 h)**

- [ ] En el guion, una respuesta de máximo dos frases y con cifra a cada una de estas preguntas probables:
  1. ¿Qué parte de los datos es real y cuál simulada? ¿Cómo saben que la simulación es realista?
  2. ¿El resultado depende del componente simulado?
  3. ¿Por qué se parte por mes y no al azar? ¿Qué pasaría con `train_test_split`?
  4. ¿Por qué el stepwise y el escalamiento van dentro del Pipeline?
  5. ¿Los 5.096 registros son independientes? ¿Qué implica para los p-valores del stepwise?
  6. ¿Por qué MAPE? ¿Qué pasa cuando el valor observado es muy bajo?
  7. ¿Por qué el ingenuo estacional es una línea base más exigente que la media?
  8. ¿La regularización resolvió la multicolinealidad del artículo? ¿Cómo lo midieron?
  9. ¿Qué aporta KNN en un problema de regresión con series de tiempo?
  10. ¿Por qué modelar el logaritmo del índice?
  11. ¿Los predictores de la prueba son conocidos de antemano? (pronóstico ex post)
  12. ¿Cómo se compara su MAPE con el 4,89 % del artículo y por qué difiere?
- [ ] Agregar las que salgan de los ensayos.

* **Listo cuando:** las 12 preguntas tienen respuesta y responsable principal.

**Aleja · formato y ortografía final (2 h)**

- [ ] Revisar numeración de figuras y tablas, que cada una se cite en el texto antes de aparecer, portada y referencias.
- [ ] Pasar el corrector del Word a todo el documento y leerlo una vez completo en voz alta.

* **Listo cuando:** versión 3 sin errores marcados por el corrector.

**Todos · ensayo 1 (1 h, 19:00):** se presenta completo con cronómetro; quien no habla hace de profesor y pregunta del banco. Juanes anota tiempo y fallas.

### Día 19 · martes 27 de octubre

**Objetivo del día:** todo cerrado un día antes.

- [ ] **Todos · ensayo 2 (1 h):** con los cambios del ensayo 1. El que hace de profesor elige preguntas al azar y a cualquier persona, no al dueño de la sección.
- [ ] **Aleja (2 h):** exportar el informe a PDF y subirlo a `docs/informe/`; etiquetar `v2.0` en main; invitar al profesor o hacer público el repositorio, y comprobar el enlace en una ventana de incógnito o con la cuenta del profesor.
- [ ] **Julián (1 h):** correr `reproducir.py --hasta prueba` desde un clon limpio de `v2.0` por última vez.
- [ ] **Juanes (1 h):** cuadre final de cifras entre el PDF y las tablas de `v2.0`.

* **Listo cuando:** PDF, etiqueta y enlace verificados; el ensayo 2 sin fallas graves.

### Día 20 · miércoles 28 de octubre · Entrega

- [ ] **Aleja:** entregar el PDF y el enlace al repositorio (etiqueta `v2.0`) por el canal del curso, antes de la hora límite, y avisar en el chat.
- [ ] **Todos (30 min):** repaso rápido del banco de preguntas si la sustentación es ese día.
- [ ] **Después de la sustentación:** cada quien anota en `docs/decisiones.md` la retroalimentación del profesor sobre su parte; es el punto de partida de la entrega final.

## Lo que no está resuelto todavía

- Fecha y duración de la sustentación: ajustar el bloque 7 cuando se sepa.
- Los enlaces directos de `interes.xlsx` y del PDF de Fedesarrollo en `data/raw/FUENTES.md` (Julián).

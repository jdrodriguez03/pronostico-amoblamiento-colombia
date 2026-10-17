# Diccionario de datos

Dueño: Julián. El diccionario completo de la base está en [`data/README.md`](../data/README.md).
Aquí se agregan las transformaciones que se decidan en la Entrega 2 (logaritmos, escalado), con su porqué.

**Estado:** propuesta del día 2 (10 de octubre). Se confirma con Juanes el día 5, porque él las
implementa dentro de `crear_preprocesador()`. Toda transformación va **dentro del Pipeline**
(regla 2): lo que se aprende de los datos (la media y la desviación del escalado) se calcula solo con
los meses de entrenamiento de cada fold.

## Transformaciones propuestas

| Columna | Transformación en el Pipeline | Por qué |
| --- | --- | --- |
| `indice_ventas_real` (objetivo) | Logaritmo, con `TransformedTargetRegressor` (ya decidido el 9 de octubre) | El índice es L × F × η: en logaritmo los factores se suman, que es lo que asume la regresión lineal |
| `area_vivienda_m2_rez1` | **Logaritmo** y luego estandarizar | Es la más asimétrica (1,58 → −0,54 con log) y varía de ≈ 6 mil a ≈ 1,1 millones de m² entre dominios y meses: un efecto de proporciones, no de diferencias absolutas. Todos sus valores son positivos |
| `importaciones_muebles_usd_miles_rez1` | Solo estandarizar. **No** aplicar logaritmo | Ya es casi simétrica (0,09); con log pasa a −0,88, es decir, empeora |
| `tasa_consumo_rez1` | Solo estandarizar | Con log la asimetría baja de 1,34 a 1,14: sigue alta y no justifica perder la lectura directa en puntos porcentuales |
| `inflacion_anual_rez1` | Solo estandarizar | Asimetría moderada (0,58); el log mejora poco (−0,26) y una tasa en porcentaje no se interpreta bien en logaritmo |
| `icc_rez1` | Solo estandarizar | Toma valores negativos (hasta −41,3): el logaritmo no se puede aplicar |
| `covid` | Pasa sin cambios | Ya es binaria (1 de abril a agosto de 2020) |

Las categóricas (`mes`, `departamento`, `linea`) llevan codificación one-hot con `drop="first"`; eso
lo define Juanes en el preprocesamiento.

**Sobre la guía.** El plan del día 2 sugería "logaritmo de importaciones y de área de vivienda" como
ejemplo. Los datos apoyan el de vivienda pero no el de importaciones, así que se propone distinto.
Si más adelante se quiere probar el log de importaciones, se decide con validación cruzada, no por
adelantado.

## Evidencia (solo entrenamiento, escenario central)

Generada por `notebooks/02_calidad_predictores.ipynb`:

- Asimetría con y sin logaritmo: `results/tablas/asimetria_transformaciones.csv`.
- Descriptiva de los 10 predictores: `results/tablas/descriptiva_predictores.csv`.
- VIF inicial de los 5 predictores rezagados: `results/tablas/vif_inicial.csv`.

| Predictor rezagado | VIF |
| --- | --- |
| `icc_rez1` | 1,83 |
| `inflacion_anual_rez1` | 6,28 |
| `tasa_consumo_rez1` | 5,60 |
| `importaciones_muebles_usd_miles_rez1` | 2,51 |
| `area_vivienda_m2_rez1` | 1,04 |

Ninguno llega a 10 (el artículo tenía 16,16 y 23,18). Inflación y tasa de consumo son los más
altos porque están correlacionadas (0,85); aplicar logaritmo a vivienda e importaciones casi no
cambia los VIF, así que las transformaciones se justifican por la forma de la distribución y no por
multicolinealidad.

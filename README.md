# Pronóstico de ventas de muebles y artículos del hogar en Colombia

[![CI](https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia/actions/workflows/ci.yml/badge.svg)](https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia/actions/workflows/ci.yml)
## Acceso al streamlit:
[![Streamlit App](https://streamlit.io)](https://pronostico-amoblamiento-colombia-wnijbu3tvw8nuwtmczyqxt.streamlit.app/)

Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia · 2026

> **¿Pueden las variables macroeconómicas pronosticar las ventas minoristas de muebles en Colombia mejor que el simple patrón estacional?**
> Reproducimos con datos del DANE el planteamiento de İnce y Taşdemir (2024), que respondieron esa pregunta para Estados Unidos con regresión lineal múltiple, y lo ampliamos con regularización (Ridge, Lasso) y KNN.

| Entrega | Contenido | Estado |
| --- | --- | --- |
| 1 | Comprensión del artículo, planteamiento y construcción de la base | ✅ Entregada y corregida ([informe](reports/entrega_1/)) |
| 2 | Flujo sin fuga, comparación de modelos y validación cruzada | ✅ Entregada el 28 de octubre de 2026 |
| Final | Correcciones, rezagos realistas, evaluación única en prueba y app de Streamlit | 🚧 En curso, entrega el 11 de noviembre de 2026 |

## El problema en cifras

- **Variable objetivo:** índice de ventas reales (base 2019 = 100) de la Encuesta Mensual de Comercio del DANE.
- **Panel:** 7 dominios departamentales × 8 líneas de bienes del hogar × 91 meses (enero 2019 – julio 2026) = **5.096 registros**.
- **Lo que se evalúa:** la línea 8, *electrodomésticos y muebles*, agregada a nivel nacional, con MAPE (y MAD y MSD como contraste), igual que en el artículo.
- **Predictores:** confianza del consumidor (Fedesarrollo), inflación anual (IPC), tasa de interés de consumo (Superfinanciera), importaciones de muebles (capítulo 94) y área aprobada para vivienda por departamento (ELIC). Entran con el rezago con que realmente se publican el día del pronóstico: tasa de consumo de t−1, ICC e inflación de t−2, importaciones y licencias de vivienda de t−3 ([`docs/disponibilidad_datos.md`](docs/disponibilidad_datos.md)), más una variable de pandemia (abril a agosto de 2020).

### Qué parte de los datos es real

El índice de cada departamento y línea se construye como **L × F × η**:

| Componente | Origen | Qué aporta |
| --- | --- | --- |
| **L** | Real (DANE, serie nacional de cada línea) | Tendencia y estacionalidad de cada línea |
| **F** | Real (DANE, factor regional del grupo CIIU 4741–4759) | Diferencias entre departamentos |
| **η** | Simulado (AR(1), ρ = 0,6, semilla 42) | Variación propia de cada serie, que el DANE no publica |

η se reescala para que el agregado nacional reproduzca **exactamente** la serie real, así que la evaluación principal no depende del componente simulado. Para comprobarlo, η se genera en tres escenarios de ruido: **bajo (2 %)**, **central (7 %, el principal)** y **alto (13 %)**. Las cinco pruebas de la simulación están en [`results/verificacion.md`](results/verificacion.md): se cumplen 3 de 5 (fallan la correlación entre líneas y las participaciones departamentales), así que lo regional vale solo dentro de los escenarios ([`docs/informe/procedencia_datos.md`](docs/informe/procedencia_datos.md)).

## Cómo se modela

```
base_final_{escenario}.csv
   ├─ entrenamiento: feb 2019 – jul 2025 ──► validación cruzada temporal (5 folds de 6 meses, partición por mes)
   │                                         Pipeline: codificar y escalar → stepwise o penalización → modelo
   │                                         todo se ajusta dentro de cada fold
   └─ prueba: ago 2025 – jul 2026 ─────────► se abre una sola vez, con el modelo ya elegido
```

| Modelos | Líneas base |
| --- | --- |
| MLR con stepwise (α = 0,25, como el artículo) · MLR completa · Ridge · Lasso · KNN | Predictor de la media · Ingenuo estacional (mismo mes del año anterior) |

El objetivo se modela en logaritmo. La especificación quedó congelada antes de abrir la prueba ([`docs/especificacion_final.md`](docs/especificacion_final.md)). Las decisiones y su justificación están en [`docs/decisiones.md`](docs/decisiones.md) y la estrategia de validación en [`docs/validacion.md`](docs/validacion.md).

## Inicio rápido

Requiere **Python 3.12** y, para leer el PDF de Fedesarrollo, **poppler** (`pdftotext`). Para instalarlo: `sudo apt install poppler-utils` en Linux, `brew install poppler` en Mac, o en Windows descargar poppler y agregar su carpeta `bin` al PATH.

```bash
git clone https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia.git
cd pronostico-amoblamiento-colombia
python -m venv .venv
source .venv/Scripts/activate        # Mac o Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest                               # comprueba que todo está en orden
```

**Reconstruir la base desde los anexos originales:**

```bash
python src/simulation/simular_base.py   # variable objetivo → data/interim/ y results/verificacion.md
python src/features/predictores.py      # predictores y unión → data/processed/base_final_*.csv
git status                              # no debe cambiar nada: la base es reproducible byte a byte
```

**Usar la base en código:**

```python
from src.datos.cargar import cargar_entrenamiento

df = cargar_entrenamiento("central")    # valida el contrato y devuelve feb 2019 – jul 2025
```

## Entrenar y comparar modelos

Desde la raíz del repositorio, ejecuta la validación cruzada de un escenario o de los tres:

```bash
python scripts/entrenar.py --escenario central
python scripts/entrenar.py --escenario bajo
python scripts/entrenar.py --escenario alto
python scripts/entrenar.py --escenario todos
```

Si omites `--escenario`, se usa `central`. El script compara los cinco modelos candidatos y las dos líneas base con validación temporal por mes. Los ajustes y los folds trabajan con las fechas que devuelve `cargar_entrenamiento` (anteriores a agosto de 2025); este comando no ejecuta la evaluación del conjunto final.

Las tablas oficiales quedan en `results/tablas/`:

- `cv_resultados.csv`: métricas e hiperparámetros por escenario, modelo y fold.
- `comparacion_cv.csv`: resumen por escenario y modelo.
- `comparacion_pareada.csv`: diferencias de MAPE por fold frente a los modelos de referencia.

Los modelos ajustados se guardan en `results/modelos/`. Para comprobar el flujo con la base pequeña y una rejilla reducida, usa:

```bash
python scripts/entrenar.py --rapido
```

La corrida rápida escribe sus tablas y modelos en `results/modelos/rapido/`; esos resultados son de prueba del flujo, no sustituyen las tablas oficiales.

## Resultado en prueba

La prueba (agosto 2025 – julio 2026) se abrió una sola vez. Por la regla escrita antes de abrirla, el modelo final es **KNN** (k = 3): MAPE nacional de **19,8 %** (IC 95 % por bloques: 16,5 – 22,8 %), frente a 25,7 % de Ridge. **Ningún modelo superó al ingenuo estacional (16,1 %)**: todos subestiman los 12 meses porque el mercado creció a niveles que los predictores rezagados no anticipaban. Detalle en `results/tablas/tabla_final.csv` y en [`docs/decisiones.md`](docs/decisiones.md).

## Cómo correr la app

```bash
pip install -r requirements.txt     # incluye streamlit
streamlit run app/app.py            # desde la raíz del repositorio; abre http://localhost:8501
```

La app pronostica el índice de ventas reales (2019 = 100) para el ámbito (nacional o un departamento), la línea y el mes elegidos. Muestra los datos de entrada con el mes de cada dato, la referencia del mismo mes del año anterior, el rango p10–p90 y un panel de drift de los predictores. Entrena el modelo final con febrero 2019 – julio 2025 al abrirse (unos segundos) y no evalúa con la prueba.

## Reproducir todo

```bash
python scripts/reproducir.py --hasta datos     # base: simulación + predictores (menos de 1 minuto)
python scripts/reproducir.py --hasta cv        # + validación cruzada de los 3 escenarios
python scripts/reproducir.py --hasta prueba    # + evaluación en prueba (--forzar; solo para comprobar un clon limpio)
```

Al final imprime el MD5 de cada archivo de `data/` y `results/tablas/` (sin tener en cuenta el fin de línea, para que coincida en Windows, Mac y Linux) y el tiempo total. La corrida completa tarda [COMPLETAR] minutos.

Tablas en `results/tablas/` y figuras en `results/figuras/`.

## Estructura

```
data/
  raw/          8 anexos originales y FUENTES.md (enlaces y fechas de descarga)
  interim/      bases por escenario, predictores y archivos de auditoría
  processed/    base_final_{bajo,central,alto}.csv: la que entra a los modelos
  README.md     diccionario de datos
src/
  simulation/   construcción de la variable objetivo
  features/     construcción de los predictores
  config.py     rutas, columnas, fechas y semilla (fuente única)
  datos/        contrato de la base y carga
  modelos/      preprocesamiento, Pipeline, stepwise, candidatos, líneas base
  evaluacion/   partición temporal, métricas, agregación nacional, interpretación
  despliegue/   lógica de la app (sin interfaz), probada en tests/test_predecir.py
app/            app de Streamlit (app.py)
scripts/        entrenar, evaluar en prueba y reproducir todo
notebooks/      verificación, predictores, comparación, interpretación, escenarios, patrón de errores
results/        verificación de la base, tablas y figuras
reports/        informe de la entrega 1 y referencias
docs/           guías del equipo, decisiones, validación
tests/          pruebas automáticas (corren en cada PR)
```

## Equipo

| Integrante | Responde por |
| --- | --- |
| Alejandra Cortés | Variable objetivo y simulación, verificación de la base, auditoría de fuga |
| Julián Rodríguez | Predictores y fuentes, modelos candidatos, interpretación |
| Juan Esteban Acosta Morales | Validación temporal, Pipeline, métricas y resultados |

Cómo trabajamos: [guía del repositorio](docs/guia_repositorio.md) · [plan día por día](docs/guia_dia_a_dia.md).

## Referencias

- İnce, M. N. y Taşdemir, Ç. (2024). Forecasting Retail Sales for Furniture and Furnishing Items through the Employment of Multiple Linear Regression and Holt–Winters Models. *Systems*, 12(6), 219. https://doi.org/10.3390/systems12060219 (acceso abierto, CC BY 4.0).
- DANE: Encuesta Mensual de Comercio, Licencias de Construcción, Importaciones e IPC. Fedesarrollo: Encuesta de Opinión del Consumidor. Superintendencia Financiera: interés bancario corriente. Enlaces y fechas en [`data/raw/FUENTES.md`](data/raw/FUENTES.md).

Se usó IA generativa como apoyo para organizar el trabajo, revisar código y redactar borradores y código de apoyo (app, análisis de errores), que el equipo revisó y ajustó. Todas las decisiones técnicas fueron tomadas y pueden ser sustentadas por el equipo.

## Licencia

El código, los notebooks y los documentos propios del equipo están bajo licencia [MIT](LICENSE). Los anexos de `data/raw/` conservan los términos de uso de sus fuentes (DANE, Fedesarrollo, Superintendencia Financiera), el artículo de İnce y Taşdemir su licencia CC BY 4.0, y el material del curso en `reports/referencias/` pertenece a sus autores.

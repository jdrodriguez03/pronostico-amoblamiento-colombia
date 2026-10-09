# Pronóstico de ventas de muebles y artículos del hogar en Colombia

[![CI](https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia/actions/workflows/ci.yml/badge.svg)](https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia/actions/workflows/ci.yml)

Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia · 2026

> **¿Pueden las variables macroeconómicas pronosticar las ventas minoristas de muebles en Colombia mejor que el simple patrón estacional?**
> Reproducimos con datos del DANE el planteamiento de İnce y Taşdemir (2024), que respondieron esa pregunta para Estados Unidos con regresión lineal múltiple, y lo ampliamos con regularización (Ridge, Lasso) y KNN.

| Entrega | Contenido | Estado |
| --- | --- | --- |
| 1 | Comprensión del artículo, planteamiento y construcción de la base | ✅ Entregada y corregida ([informe](reports/entrega_1/)) |
| 2 | Flujo sin fuga, comparación de modelos y evaluación en prueba | 🚧 En curso, entrega el 28 de octubre de 2026 |

## El problema en cifras

- **Variable objetivo:** índice de ventas reales (base 2019 = 100) de la Encuesta Mensual de Comercio del DANE.
- **Panel:** 7 dominios departamentales × 8 líneas de bienes del hogar × 91 meses (enero 2019 – julio 2026) = **5.096 registros**.
- **Lo que se evalúa:** la línea 8, *electrodomésticos y muebles*, agregada a nivel nacional, con MAPE (y MAD y MSD como contraste), igual que en el artículo.
- **Predictores:** confianza del consumidor (Fedesarrollo), inflación anual (IPC), tasa de interés de consumo (Superfinanciera), importaciones de muebles (capítulo 94) y área aprobada para vivienda por departamento (ELIC). Todos entran con un mes de rezago, más una variable de pandemia (abril a agosto de 2020).

### Qué parte de los datos es real

El índice de cada departamento y línea se construye como **L × F × η**:

| Componente | Origen | Qué aporta |
| --- | --- | --- |
| **L** | Real (DANE, serie nacional de cada línea) | Tendencia y estacionalidad de cada línea |
| **F** | Real (DANE, factor regional del grupo CIIU 4741–4759) | Diferencias entre departamentos |
| **η** | Simulado (AR(1), ρ = 0,6, semilla 42) | Variación propia de cada serie, que el DANE no publica |

η se reescala para que el agregado nacional reproduzca **exactamente** la serie real, así que la evaluación principal no depende del componente simulado. Para comprobarlo, η se genera en tres escenarios de ruido: **bajo (2 %)**, **central (7 %, el principal)** y **alto (13 %)**. Las cinco pruebas de la simulación están en [`results/verificacion.md`](results/verificacion.md).

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

El objetivo se modela en logaritmo. Las decisiones y su justificación están en [`docs/decisiones.md`](docs/decisiones.md) y la estrategia de validación en [`docs/validacion.md`](docs/validacion.md).

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
scripts/        entrenar, evaluar en prueba y reproducir todo
notebooks/      verificación, predictores, comparación, interpretación, escenarios
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

Se usó IA generativa como apoyo para organizar el trabajo y revisar código. Todas las decisiones técnicas fueron tomadas y pueden ser sustentadas por el equipo.

## Licencia

El código, los notebooks y los documentos propios del equipo están bajo licencia [MIT](LICENSE). Los anexos de `data/raw/` conservan los términos de uso de sus fuentes (DANE, Fedesarrollo, Superintendencia Financiera), el artículo de İnce y Taşdemir su licencia CC BY 4.0, y el material del curso en `reports/referencias/` pertenece a sus autores.

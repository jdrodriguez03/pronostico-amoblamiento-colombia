# Pronóstico de ventas de bienes del hogar en Colombia

Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia · 2026

**Integrantes:** Juan Esteban Acosta Morales, Julián Rodríguez, Alejandra Cortés

## Resumen

Reproducimos con datos colombianos el objetivo de İnce y Taşdemir (2024): pronosticar las ventas minoristas de muebles y artículos del hogar con regresión lineal múltiple y variables macroeconómicas.
La variable objetivo es el índice de ventas reales (base 2019 = 100) de la Encuesta Mensual de Comercio del DANE, en un panel de 7 dominios departamentales × 8 líneas del hogar × 91 meses (enero de 2019 a julio de 2026; 5.096 registros).
El índice combina dos componentes reales del DANE (la serie nacional de cada línea y el factor regional del grupo CIIU 4741–4759) con un único componente simulado, η, que se genera en tres escenarios de ruido (2 %, 7 % y 13 %) para analizar la sensibilidad.
Los predictores son la confianza del consumidor, la inflación, la tasa de consumo, las importaciones de muebles y el área aprobada para vivienda (departamental), cada uno también con un mes de rezago.
Se comparan la MLR con stepwise, Ridge, Lasso y KNN frente a dos líneas base, con validación cruzada temporal y sin fuga de datos.

## Estructura

```
data/
  raw/          anexos originales (DANE, Superfinanciera, Fedesarrollo), sin modificar
  interim/      bases por escenario, predictores y archivos de auditoría
  processed/    base_final_{bajo,central,alto}.csv: la base para modelar (principal: central)
  README.md     diccionario de datos
src/
  simulation/   simular_base.py: construye la variable objetivo y la verificación
  features/     predictores.py: construye los predictores y los une con la base
  datos/ modelos/ evaluacion/ config.py   módulos de la Entrega 2 (en construcción)
results/        verificacion.md: las 5 pruebas de la base en los 3 escenarios
reports/
  entrega_1/    informe corregido y cambios por la retroalimentación
  referencias/  artículo base, rúbrica y catálogo de artículos
notebooks/ scripts/ tests/ docs/   flujo de modelamiento de la Entrega 2
```

## Cómo reproducir la base

Desde la raíz del repositorio, con Python 3.12:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Mac o Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python src/simulation/simular_base.py
python src/features/predictores.py
```

1. `simular_base.py` lee los dos anexos de la EMC en `data/raw/`, escribe `base_{bajo,central,alto}.csv`, `componentes_regionales.csv` y `lineas_nacionales.csv` en `data/interim/`, y la verificación en `results/verificacion.md`.
2. `predictores.py` lee los insumos de `data/raw/` y las bases de `data/interim/`, escribe los `predictores_*.csv` en `data/interim/` y las `base_final_*.csv` en `data/processed/`.

La semilla es fija (42): con los mismos anexos, el resultado es idéntico.

**Requisito adicional:** `predictores.py` lee el ICC de Fedesarrollo desde un PDF con `pdftotext`, que viene en **poppler-utils** (no se instala con pip). Linux: `sudo apt install poppler-utils`. Mac: `brew install poppler`. Windows: descargar poppler y agregar su carpeta `bin` al PATH.

## Diccionario de datos

Columnas, fuentes, niveles y notas de cada base: ver [`data/README.md`](data/README.md).

Tres advertencias que no son errores:

- Las tres bases (bajo, central, alto) son escenarios de ruido a propósito; solo cambia `indice_ventas_real`.
- `componentes_regionales.csv` y `lineas_nacionales.csv` son de auditoría: **no se usan como predictores**, porque sería fuga de información.
- `area_vivienda_m2_rez1` está vacía en enero de 2019 (56 filas) a propósito.

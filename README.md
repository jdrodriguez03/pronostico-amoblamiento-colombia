# Pronóstico de ventas de amoblamiento y hogar en Colombia

Proyecto Integrador · Machine Learning I · Universidad Externado de Colombia · 2026
Integrantes: Juan Esteban Acosta Morales, Julián Rodríguez, Alejandra Cortés

Reproducimos, con datos colombianos, el objetivo de İnce y Taşdemir (2024), *Forecasting Retail
Sales for Furniture and Furnishing Items through the Employment of Multiple Linear Regression and
Holt–Winters Models* (Systems, 12(6), 219): pronosticar el índice mensual de ventas reales de
bienes de amoblamiento y hogar a partir de variables macroeconómicas. Se compara la regresión
lineal múltiple con stepwise (el método del artículo) con Ridge, Lasso y KNN, frente a dos líneas
base, sobre un panel de 7 dominios departamentales × 8 líneas × 91 meses (5.096 registros)
construido con la Encuesta Mensual de Comercio del DANE.

> Estado: **Entrega 2 en construcción** (datos, modelamiento y validación). Ver `docs/`.

## Instalación (una vez)

Requiere Python 3.12 y Git. En Windows, usar Git Bash.

```bash
git clone https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia
cd pronostico-amoblamiento-colombia
python -m venv .venv
source .venv/Scripts/activate        # Mac o Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # para que "from src..." funcione en todas partes
pytest                                # debe salir en verde
```

## Cómo reproducir

```bash
python scripts/reproducir.py --hasta datos    # data/raw -> data/processed
python scripts/reproducir.py --hasta cv       # + validación cruzada de todos los modelos
python scripts/reproducir.py --hasta prueba   # + evaluación única en el conjunto de prueba
```

Las tablas quedan en `resultados/tablas/` y las figuras en `resultados/figuras/`.
(Se completa a medida que avanzan los bloques.)

## Estructura

| Carpeta | Para qué |
| --- | --- |
| `src/datos/` | Arma los datos: lectura del DANE, simulación, panel, predictores |
| `src/modelos/` | Preprocesamiento, Pipeline, líneas base, stepwise, candidatos |
| `src/evaluacion/` | Validación cruzada por mes, métricas, agregación nacional, interpretación |
| `scripts/` | Programas que se corren desde la terminal |
| `data/raw/` | Archivos originales (fuentes en `data/raw/FUENTES.md`) |
| `data/processed/` | Lo que generan los scripts |
| `notebooks/` | Análisis y figuras; importan de `src/` |
| `resultados/` | Tablas y figuras del informe |
| `tests/` | Pruebas automáticas (`pytest`) |
| `docs/` | Guías del equipo, decisiones, diccionario, borradores del informe |

Antes de trabajar, leer la guía rápida del repositorio y la guía día por día (documentos compartidos del equipo).

## Uso de IA generativa

Ver `docs/uso_ia.md`.

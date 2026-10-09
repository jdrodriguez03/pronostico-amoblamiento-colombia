# Bitácora de decisiones metodológicas

Archivo compartido: **solo se agregan entradas al final**; nadie edita las de otro.
Formato: fecha · quién · decisión · por qué (una o dos líneas).

| Fecha | Quién | Decisión | Por qué |
| --- | --- | --- | --- |
| Entrega 1 | Equipo | Panel de 7 dominios × 8 líneas × 91 meses (5.096 registros) | Ninguna fuente real de ventas de muebles alcanza 5.000 registros; el objetivo del artículo se evalúa sobre la línea 8 agregada a nivel nacional |
| Entrega 1 | Equipo | Índice = L (real) × F (real) × η (simulado, AR(1), ρ = 0,6) | Minimiza la parte simulada; η se ajusta para conservar exactamente la serie nacional |
| Entrega 1 | Equipo | Tres escenarios de η: 2 %, 7 % y 13 %, semilla 42 | Fijados antes de modelar; las conclusiones se reportan en los tres |
| Entrega 1 | Equipo | Pesos departamentales base 2019 desde las contribuciones del DANE | Reproducen el índice nacional de las actividades 474/475 (error 8,7e-14 %) |
| Entrega 1 | Equipo | `covid` = 1 de abril a agosto de 2020 | Aislamiento preventivo obligatorio nacional (25-mar a 31-ago-2020) |
| Entrega 1 | Equipo | ICC solo nacional | El histórico de Fedesarrollo descargado no trae el ICC por ciudad; la única variación regional real es la vivienda |
| Entrega 1 | Equipo | Importaciones del capítulo 94 completo | El DANE no publica las partidas 9401/9403/9404 en sus anexos; incluye colchones, lámparas y prefabricados |
| Entrega 1 | Equipo | Columnas `_rez1` (un mes de rezago) | Al pronosticar el mes t aún no se conocen sus datos económicos |
| Entrega 1 | Equipo | `area_vivienda_m2_rez1` vacía en enero de 2019 | Antes de 2019 la cobertura de ELIC no es comparable |
| Entrega 1 | Equipo | Se excluyen población e índice de precios al productor | Fueron la fuente de multicolinealidad en el artículo (VIF 16,16 y 23,18) |
| 2026-10-08 | Julián | Repositorio con la base de la Entrega 1 verificada | Los scripts reproducen byte a byte todas las bases y la verificación |
| Propuesta día 1 | — | `USAR_REZAGO = True` y se descarta enero de 2019 | Pronóstico realista sin imputar; queda en `config.py`, se confirma en el arranque |
| 2026-10-09 | Julián | Confirmado: predictores rezagados (`USAR_REZAGO = True`), sin enero de 2019; 78 meses de entrenamiento | Al pronosticar el mes t aún no se conocen sus datos económicos; descartar el único mes sin rezago evita imputar |
| 2026-10-09 | Julián | Objetivo en logaritmo con `TransformedTargetRegressor` (`LOG_OBJETIVO = True`) | El índice es L × F × η: en logaritmo los factores se suman, que es lo que asume la regresión lineal; las métricas se calculan en la escala original |
| 2026-10-09 | Julián | KNN para regresión como cuarto modelo, junto a MLR (stepwise y completa), Ridge y Lasso | La rúbrica de la 7.3 nombra KNN y pide varios modelos; aporta un contraste no lineal |
| 2026-10-09 | Julián | El repositorio se hace público el 27 de octubre | El profesor lo revisa sin necesitar invitación; hasta entonces sigue privado |

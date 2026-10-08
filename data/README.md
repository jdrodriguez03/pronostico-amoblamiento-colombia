# Base de datos — Proyecto Integrador ML I

## Archivos

| Archivo | Contenido |
|---|---|
| `base_bajo.csv` | Base con η = 2 % (escenario bajo) |
| `base_central.csv` | Base con η = 7 % (escenario central, el principal) |
| `base_alto.csv` | Base con η = 13 % (escenario alto) |
| `componentes_regionales.csv` | Pesos, factor regional e índices del grupo 4741–4759 por departamento y mes. Solo para auditoría: **no usar como predictores** (sería fuga de información) |
| `lineas_nacionales.csv` | Índices nacionales reales de las 8 líneas. Solo para auditoría y para evaluar la línea 8 a nivel nacional |
| `verificacion.md` | Resultado de las 5 pruebas de verificación en los 3 escenarios |
| `simular_base.py` | Código que genera todo lo anterior a partir de los dos anexos del DANE |

## Columnas de las bases

| Columna | Descripción |
|---|---|
| `fecha` | Primer día del mes (2019-01-01 a 2026-07-01) |
| `anio`, `mes` | Año y mes |
| `departamento` | Uno de los 7 dominios de la EMC |
| `linea_codigo` | Código de la línea de mercancía en la EMC (4, 8, 9, 10, 11, 12, 14, 15) |
| `linea` | Nombre de la línea |
| `es_muebles` | 1 si es la línea 8 (electrodomésticos y muebles), 0 si no |
| `indice_ventas_real` | **Variable objetivo**: índice de ventas reales, base 2019 = 100 |

Cada base tiene 5.096 filas (7 departamentos × 8 líneas × 91 meses).

## Cómo regenerarla

```
python simular_base.py anex-EMC-ComercioMinorista-jul2026.xlsx anex-EMC-ComercioalpormenorDep-jul2026.xlsx salida
```

Con la misma semilla (42) y los mismos anexos, el resultado es idéntico.

## Cómo evaluar la línea de muebles a nivel nacional

Agregar las predicciones de la línea 8 de los 7 departamentos con la columna `peso_base2019` de `componentes_regionales.csv` (promedio ponderado). Por construcción, ese agregado reproduce exactamente la serie nacional real.

## Base final con predictores (`base_final_bajo.csv`, `base_final_central.csv`, `base_final_alto.csv`)

Mismas filas y columnas de arriba, más los predictores. Se generan con `predictores.py`.

| Columna | Descripción | Fuente | Nivel |
|---|---|---|---|
| `icc` | Índice de Confianza del Consumidor (balance) | Fedesarrollo, EOC histórico | Nacional |
| `inflacion_anual` | Variación anual del IPC total, en % | DANE, IPC series de empalme | Nacional |
| `tasa_consumo` | Interés bancario corriente, crédito de consumo y ordinario, % efectivo anual | Superfinanciera | Nacional |
| `importaciones_muebles_usd_miles` | Importaciones del capítulo 94 del arancel, miles de USD CIF | DANE, importaciones por capítulo | Nacional |
| `area_vivienda_m2` | Área aprobada para vivienda (destino = vivienda), m² | DANE, ELIC por municipio, agregado a los 7 dominios | **Departamental** |
| `covid` | 1 entre abril y agosto de 2020 (aislamiento preventivo obligatorio nacional) | Construida | Nacional |
| `*_rez1` | Mismo predictor con un mes de rezago | — | — |

Notas:
- Las columnas `_rez1` existen porque, al pronosticar un mes, todavía no se conocen los datos económicos de ese mes. La decisión de usar valores del mismo mes o rezagados se toma en la Segunda Entrega.
- `area_vivienda_m2_rez1` está vacía en enero de 2019 (56 filas): antes de 2019 la cobertura de ELIC era menor y no es comparable.
- Verificación: la suma de `area_vivienda_m2` de los 7 dominios coincide exactamente con el total nacional oficial de ELIC.
- El ICC de Fedesarrollo descargado solo trae el agregado nacional (no por ciudad). Por eso el único predictor con variación regional real es la vivienda.
- Las importaciones son del capítulo 94 completo: incluye muebles, pero también colchones, lámparas y construcciones prefabricadas. El DANE no publica las partidas en sus anexos.

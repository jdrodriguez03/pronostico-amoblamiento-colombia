# Disponibilidad real de los predictores

> Dueño: Julián. Responde a la corrección 2 de la Entrega 2.

## Fecha de emisión

El pronóstico del mes t se emite el **primer día hábil del mes t** y solo usa lo publicado hasta ese día. Para cada predictor, k es el menor número de meses tal que el dato de t−k ya está publicado en la fecha de emisión.

## Rezagos resultantes

| Predictor | Entidad | Días entre el cierre del mes y la publicación (mín · mediana · máx) | Último mes disponible el 1 de marzo | k |
| --- | --- | --- | --- | --- |
| `icc` | Fedesarrollo (Encuesta de Opinión del Consumidor) | 9 · 11,5 · 16 | Enero | 2 |
| `inflacion_anual` | DANE (IPC) | 5 · 5,5 · 8 | Enero | 2 |
| `tasa_consumo` | Superintendencia Financiera (Formato 88, semanal) | 1 · 4 · 7 | Febrero (semanas publicadas) | 1 |
| `importaciones_muebles_usd_miles` | DANE con registros de la DIAN | 45 · 50 · 52 | Diciembre | 3 |
| `area_vivienda_m2` | DANE (ELIC) | 45 · 45 · 47 | Diciembre | 3 |

`covid`, `mes`, `departamento` y `linea` se conocen de antemano y no llevan rezago.

El vector `[2, 2, 1, 3, 3]` se comprobó con tres fechas de emisión (1 de marzo, 1 de agosto y 1 de noviembre) y no cambia. Las fechas de publicación salen de los calendarios de difusión del DANE y de los boletines de Fedesarrollo y la Superintendencia Financiera (enlaces en `data/raw/FUENTES.md`); algunas se infirieron del ciclo habitual de publicación y no de un boletín individual.

## Supuesto declarado

**Tasa de consumo (k = 1).** El primer día del mes aún puede faltar la última semana del mes anterior, así que el dato de t−1 es un promedio de las semanas ya publicadas. La tasa se mueve lentamente, por lo que la diferencia frente al promedio completo es pequeña. La alternativa conservadora sería k = 2.

## Texto para el informe

El pronóstico del mes t se emite el primer día hábil de ese mes y solo usa la información publicada hasta entonces. Según los calendarios de difusión del DANE, Fedesarrollo y la Superintendencia Financiera, la tasa de consumo entra con un mes de rezago (promedio de las semanas ya publicadas), el ICC y la inflación con dos, e importaciones y licencias con tres, porque se publican unos 45 a 50 días después del cierre del mes. El vector de rezagos no cambia con el mes de emisión.

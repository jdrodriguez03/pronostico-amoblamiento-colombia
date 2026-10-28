# 7.4 Interpretación preliminar del modelo

El modelo preliminar es Ridge, reentrenado con todos los meses de entrenamiento. Sus coeficientes
están estandarizados, así que cada uno se lee como el cambio porcentual del índice de ventas cuando
el predictor sube una desviación estándar y las demás variables se mantienen fijas. Es una
asociación condicional, no un efecto causal, y como Ridge encoge los coeficientes hacia cero, lo
que importa es el signo y el orden de magnitud más que el valor exacto. Tampoco usamos los p-valores
del stepwise como prueba de que una variable predice: con 56 registros por mes que comparten los
mismos indicadores nacionales, esos p-valores son optimistas.

Tabla 1. Variables del modelo preliminar (Ridge, escenario central)

| Variable | Efecto por una desviación estándar | ¿Signo esperado? |
| --- | --- | --- |
| Importaciones de muebles (unos 7.500 miles de USD) | +7,5 % | Sí |
| Inflación (3,6 puntos) | +5,1 % | No |
| Tasa de interés de consumo (4,1 puntos) | −2,1 % | Sí |
| Área licenciada para vivienda (el área se multiplica por 2,5) | +1,6 % | Sí |
| Confianza del consumidor (9,7 puntos) | +0,5 % | Cerca de cero |

La variable que más pesa es la importación de muebles del mes anterior, con el signo esperado, y es
también la que Lasso conserva en todas las particiones. La tasa de consumo y la vivienda tienen el
signo esperado y efectos pequeños. La inflación sale al revés, pero no la leemos como que la
inflación suba las ventas: la inflación alta de 2021 a 2023 coincidió con la recuperación del
consumo después de la pandemia, y además se correlaciona en 0,85 con la tasa de consumo. La
confianza del consumidor queda prácticamente en cero y su signo cambia según los meses con que se
entrena, así que no la interpretamos. En la estacionalidad, diciembre está 8,0 % y noviembre 4,4 %
por encima de enero, y abril 5,7 % por debajo. Los signos son los mismos en los tres escenarios.

Los errores no son aleatorios (Figura 1). Hasta junio de 2024 el modelo se equivoca hacia los dos
lados, entre −10 y +11 puntos del índice. Desde la segunda mitad de 2024 empieza a quedarse corto, y
desde noviembre de 2024 subestima todos los meses, entre 15 y 47 puntos (12 % a 29 %): el índice
sube de cerca de 120 a 165, mientras el pronóstico se mantiene entre 110 y 135. Por eso las dos
últimas particiones tienen los errores más altos (11,5 % y 18,7 %). No encontramos un patrón por
mes del año, aunque con dos o tres observaciones por mes no se puede descartar. En el ajuste, los
cinco meses del confinamiento tienen un error mucho mayor que el resto (MAPE de 37,9 % frente a
11,3 %, y 27,6 frente a 14,0 puntos de error absoluto medio); esos meses se conservan, marcados con
una variable de pandemia, aunque no medimos cuánto influyen en los coeficientes.

Figura 1. Errores del modelo preliminar en los 30 meses de validación, error por mes del año y error
de ajuste en los meses de confinamiento frente al resto.

Para una empresa de muebles, la lectura preliminar es que el dato de importaciones del mes anterior
es la señal más útil para anticipar las ventas, y que noviembre y diciembre piden más inventario que
el resto del año. La advertencia es que el modelo no sigue el crecimiento reciente del mercado: si
se usara para planear, en 2025 se habría quedado corto de inventario todos los meses. Si esa
tendencia continúa, el error en el periodo de prueba será mayor que el de validación.

Esto responde la pregunta planteada en la Entrega 1: con predictores correlacionados, la
regularización sí mejora la capacidad predictiva frente a la regresión lineal múltiple. Ridge baja
el MAPE de validación de 14,8 % a 9,1 %, ningún coeficiente cambia de signo entre particiones y su
variación es hasta 25 veces menor, mientras en la regresión sin penalizar el coeficiente de vivienda
cambia de signo según el periodo.

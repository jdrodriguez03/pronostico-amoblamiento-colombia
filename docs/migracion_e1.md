# Migración del lector EMC desde Entrega 1

## Lectura de los anexos EMC

En Entrega 1, `src/simulation/simular_base.py` implementaba la lectura dentro
del flujo de simulación. `leer_lineas()` leía la hoja `2.2` y devolvía las
fechas, una matriz nacional `L` y el libro de Excel abierto. `leer_departamentos()`
leía la hoja `3.2` y devolvía matrices y parámetros para la simulación:
`F`, `G`, `w` y `R`.

En Entrega 2, `src/datos/emc.py` separa la lectura del cálculo de la base:

- `leer_lineas_nacional()` lee la hoja `2.2` y devuelve una tabla larga con
  `fecha`, `linea` y `L`: una fila por mes y línea.
- `leer_departamental()` lee la hoja `3.2` y devuelve una tabla larga con
  `fecha`, `departamento` e `indice_depto`: una fila por mes y dominio para el
  grupo CIIU `4741` (4741–4759, según la descripción del módulo legado).
- `factor_regional()` calcula el factor a partir de los índices de esa tabla y
  los pesos base 2019 `PESOS_DEPTO` de `src/config.py`. Devuelve `fecha`,
  `departamento` y `factor_regional`.

El lector de Entrega 2 cierra el libro de Excel después de leer la hoja. El
legado devolvía el libro abierto; el código consumidor lo cerraba más adelante.

## Diferencia entre los pesos regionales

El cálculo de Entrega 1 en `leer_departamentos()` derivaba sus pesos a partir de
las hojas `1.1` y `2.1`: dividía contribuciones por variaciones para obtener el
peso total departamental, estimaba la participación del grupo hogar y
normalizaba esos valores en julio de 2025 usando los índices de la hoja `3.2`.
Con esos pesos calculaba `R = G @ w` y luego `F = G / R[:, None]`.

El cálculo actual en `factor_regional()` no vuelve a derivar esos pesos de las
hojas `1.1` y `2.1`: usa directamente `PESOS_DEPTO` desde `src/config.py` y
normaliza los índices departamentales con la suma ponderada de cada mes. Son
pesos almacenados con seis decimales en la configuración, a partir de los
pesos de `componentes_regionales.csv` de Entrega 1. La configuración y los
pesos internos de la simulación pueden diferir por el redondeo. La prueba de
agregación de Entrega 2 verifica el resultado con `PESOS_DEPTO`; no afirma que
sean idénticos a los pesos internos de precisión completa de Entrega 1.

## Verificación nacional de las actividades 474 y 475

El anexo nacional publica las actividades 474 y 475 por separado en la hoja
2.4. El anexo departamental publica en la hoja 3.2 un índice único para el
grupo 4741–4742–4751–4752–4753–4754–4755–4759 en cada dominio. No se encontró
en los anexos una serie nacional para ese grupo combinado ni coeficientes
oficiales fijos para combinar las series 474 y 475.

En Entrega 1, `src/simulation/simular_base.py` formaba el agregado del grupo
como `R = G @ w`, con pesos calculados desde las hojas departamentales 1.1 y
2.1 y los índices de julio de 2025. Luego ajustaba por mínimos cuadrados
coeficientes `b` para las series nacionales 474 y 475 (`C @ b`) usando los
mismos meses que evaluaba. El `8,7e-14 %` de `results/verificacion.md` es el
error relativo máximo dentro de ese ajuste. Como `b` no se fijaba con una
fuente independiente, esa cifra no demuestra una comparación independiente
del agregado con un índice nacional combinado previamente definido.

Por ello, la comprobación nacional 474/475 con tolerancia absoluta de ±0,1
puntos de índice queda pendiente: falta la serie nacional oficial del grupo
departamental combinado o una regla y coeficientes fijos, justificados de
forma independiente, para combinar las actividades 474 y 475. La prueba actual
de `tests/test_emc.py` conserva las comprobaciones de dimensiones, nulos,
normalización de factores y equivalencia de la lectura departamental con `G`;
no declara cumplida esa comprobación nacional.

La guía local del día 2 especifica además una comprobación distinta: agregar
las líneas 4 y 8 con `PESOS_DEPTO` y compararlas con `L_4` y `L_8` de
`lineas_nacionales.csv`, con error relativo menor que `1e-6`. Para que el
redondeo al guardar la base intermedia no exceda esa tolerancia, `simular_base.py`
escribe ahora `base_{escenario}.csv` con ocho decimales. No cambia la fórmula
de simulación ni los pesos.

## Comprobación de la lectura departamental

`tests/test_emc.py::test_lectura_departamental_coincide_con_matriz_historica`
compara los valores departamentales leídos por Entrega 2 con la matriz `G` que
produce el lector de Entrega 1. La prueba alinea explícitamente las fechas y
los dominios, comprueba que coincidan y exige igualdad exacta de los valores
para los anexos actuales. Esa prueba valida la migración de la lectura de la
hoja `3.2`; no compara los factores regionales calculados con los dos métodos
de pesos.

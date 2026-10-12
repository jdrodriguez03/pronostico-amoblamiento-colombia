# Validación: ventanas exactas de cada fold

Dueño: Juanes. Con `USAR_REZAGO = True` el entrenamiento va de febrero de 2019 a julio de 2025 (78 meses).
5 folds de ventana creciente, 6 meses de validación cada uno.

| Fold | Entrena con | Valida con |
| --- | --- | --- |
| 1 | feb 2019 – ene 2023 (48 meses) | feb 2023 – jul 2023 |
| 2 | feb 2019 – jul 2023 (54) | ago 2023 – ene 2024 |
| 3 | feb 2019 – ene 2024 (60) | feb 2024 – jul 2024 |
| 4 | feb 2019 – jul 2024 (66) | ago 2024 – ene 2025 |
| 5 | feb 2019 – ene 2025 (72) | feb 2025 – jul 2025 |
| Prueba | feb 2019 – jul 2025 (78) | ago 2025 – jul 2026 |

La implementación es `CVTemporalPorMes` en `src/evaluacion/particion.py`, y
`tests/test_particion.py` comprueba que las ventanas que genera son exactamente las de esta tabla.
Para verlas en pantalla: `CVTemporalPorMes().ventanas(cargar_entrenamiento("central"))`.

## Por qué así

**Ventana creciente.** Cada fold entrena con todos los meses anteriores a su ventana de validación y
valida con los 6 meses siguientes. Así se imita lo que hace una empresa que pronostica: usa todo el
pasado que tiene y pronostica hacia adelante. Nunca se entrena con meses posteriores a los que se
validan, porque eso sería usar el futuro para predecir el pasado. Empezar con 48 meses garantiza que
el fold 1 ya tiene cuatro años de historia, incluidos los meses de pandemia de 2020, y que cada mes del
año aparece al menos cuatro veces para estimar la estacionalidad.

**Partición por mes, no por fila.** Un mes tiene 56 filas (7 departamentos × 8 líneas) y todas
comparten los mismos predictores nacionales (ICC, inflación, tasa de consumo, importaciones) y el
mismo choque del mes. Si se repartieran las filas al azar, como hace `train_test_split`, el modelo
vería en entrenamiento otras filas del mismo mes que luego valida y el error saldría optimista: eso es
fuga de información. Por eso las 56 filas de un mes siempre quedan juntas, en entrenamiento o en
validación, y nunca se usa `shuffle=True`.

**Seis meses de validación y cinco folds.** Seis meses es un horizonte de planeación realista para
inventarios de muebles. Con 5 folds los 30 meses de validación (febrero de 2023 a julio de 2025)
cubren dos años y medio: cada mes del año se valida al menos dos veces, así que la estacionalidad pesa
igual en la comparación. Además, la variación del error entre folds mide qué tan estable es cada
modelo, que es uno de los criterios de desempate del día 11.

**El ingenuo estacional siempre tiene su t−12.** Esta línea base pronostica el mes t con el valor
del mismo mes del año anterior. Como la validación siempre empieza justo después del entrenamiento
y dura 6 meses (menos de 12), el mes t−12 de cualquier mes validado cae siempre dentro del
entrenamiento de ese fold. Lo mismo pasa en la prueba: agosto de 2025 a julio de 2026 usa agosto de
2024 a julio de 2025, que son meses de entrenamiento. La prueba `test_ingenuo_tiene_su_t_menos_12` lo
comprueba.

**La prueba queda fuera.** `CVTemporalPorMes` lanza un error si recibe algún mes desde agosto de 2025:
la validación cruzada solo trabaja con `cargar_entrenamiento()` (regla de oro 3).

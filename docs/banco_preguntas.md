# Banco de preguntas de la sustentación · Segunda entrega

Cada respuesta tiene máximo dos frases y su cifra. **Responde** es quien la contesta primero, pero
los tres deben poder responder todas: en el ensayo, quien hace de profesor le pregunta a alguien que
no es el responsable. Las cifras son del escenario central salvo que se diga otra cosa.

| # | Pregunta | Responde |
| --- | --- | --- |
| 1 | ¿Qué parte de los datos es real y cuál simulada? ¿Cómo saben que la simulación es realista? | Aleja |
| 2 | ¿El resultado depende del componente simulado? | Aleja |
| 3 | ¿Por qué se parte por mes y no al azar? ¿Qué pasaría con `train_test_split`? | Juanes |
| 4 | ¿Por qué el stepwise y el escalamiento van dentro del Pipeline? | Juanes |
| 5 | ¿Los 5.096 registros son independientes? ¿Qué implica para los p-valores del stepwise? | Julián |
| 6 | ¿Por qué MAPE? ¿Qué pasa cuando el valor observado es muy bajo? | Juanes |
| 7 | ¿Por qué hay dos líneas base, y por qué la media resultó más exigente que el ingenuo estacional? | Julián |
| 8 | ¿La regularización resolvió la multicolinealidad del artículo? ¿Cómo lo midieron? | Julián |
| 9 | ¿Qué aporta KNN en un problema de regresión con series de tiempo? ¿Por qué sin `drop`? | Julián |
| 10 | ¿Por qué modelar el logaritmo del índice? | Aleja |
| 11 | ¿Qué se pronostica exactamente: un mes o seis? ¿Los datos de t−1 están disponibles a tiempo? | Juanes |
| 12 | ¿Cómo se compara su MAPE con el 4,89 % del artículo y por qué no con el de validación? | Julián |
| 13 | ¿Por qué no evaluaron todavía en la prueba? | Juanes |
| 14 | ¿Por qué eligieron Ridge si KNN tiene menor MAPE? | Julián |
| 15 | ¿Por qué Ridge y no Lasso, si Lasso le gana a la media en más folds? | Julián |

## Respuestas

**1. Real y simulado.** La serie nacional de cada línea es real (EMC del DANE) y los predictores
también; lo simulado es solo el reparto por departamento, con parámetros estimados con datos del
DANE. La verificación lo respalda: la serie nacional se reconstruye con un error de 2,8e-14, el ruido
da 6,7 % frente al 7 % buscado y la persistencia 0,56 frente a 0,60; dos pruebas no se cumplen y lo
declaramos (correlación entre líneas 0,59 y participación ± 1,3 pts).

**2. Dependencia de la simulación.** El MAPE nacional casi no cambia entre escenarios porque la serie
nacional es real: Ridge da 9,02 %, 9,08 % y 9,14 % con ruido de 2 %, 7 % y 13 %. El orden sí cambia
en el escenario alto, donde KNN cae al cuarto puesto (10,26 %); por eso no elegimos un modelo que
dependa del escenario.

**3. Partición por mes.** Las 56 filas de un mes comparten los mismos predictores nacionales, así que
si se reparten al azar el modelo valida con meses que ya vio en entrenamiento y el error sale
optimista. Además, con `train_test_split` se entrenaría con el futuro para predecir el pasado, y en
producción solo se tiene el pasado.

**4. Todo dentro del Pipeline.** Si el escalamiento se ajustara antes de partir, la media y la
desviación incluirían los meses de validación; si el stepwise eligiera variables con toda la base,
la selección ya habría visto la validación. Dentro del Pipeline, en cada fold se ajustan solo con
sus meses de entrenamiento, y `test_fuga.py` lo vigila en cada cambio.

**5. Independencia.** No: las 56 filas de un mes comparten los 5 predictores nacionales, así que la
información efectiva está más cerca de los 78 meses de entrenamiento que de las 4.368 filas. Por eso
los p-valores del stepwise son optimistas (retiene las 6 variables con p < 1e-4) y no los usamos como
prueba de que una variable predice: la evidencia es el MAPE de validación, donde el stepwise da 15,0 %.

**6. MAPE.** El costo de un error en inventarios depende de su tamaño frente a la demanda, y el MAPE
lo expresa en porcentaje, comparable con el 4,89 % del artículo. Cuando el valor observado es muy
bajo se infla, como en abril de 2020 (índice 46,8): por eso reportamos MAD y RMSE como contraste y
la pandemia queda en entrenamiento, no en validación.

**7. Dos líneas base.** La media es el mínimo trivial del curso y el ingenuo estacional es la
referencia natural de una serie estacional; las dos marcan lo que un modelo debe superar. La media
resultó más exigente (10,3 % frente a 12,3 %) porque la serie fluctúa alrededor de un nivel y el
ingenuo arrastra el ruido de un año a otro: falla sobre todo en el fold 1 (21,3 %), que copia meses
de 2022.

**8. Multicolinealidad.** El VIF máximo es 6,28 (23,18 en el artículo) y la inflación y la tasa de
consumo se correlacionan 0,85; lo medimos con la estabilidad de los coeficientes entre los 5 folds.
En la MLR el coeficiente de vivienda cambia de signo entre folds; con Ridge ningún signo cambia, la
variación es hasta 25 veces menor y el MAPE baja de 14,8 % a 9,1 %.

**9. KNN.** Es el modelo no paramétrico del curso: pronostica con el promedio de los meses más
parecidos y no supone una relación lineal; con k = 3 da 9,0 %, el menor MAPE del central. Va sin
`drop` porque con una categoría de referencia esa categoría queda a distancia 1 de las demás y las
otras a √2 entre sí, y eso sesga los vecinos.

**10. Logaritmo.** El índice se comporta como un producto de factores (estacionalidad × nivel ×
choques), y el logaritmo lo vuelve una suma, que es lo que modela una regresión lineal. Además
reduce el peso de los meses extremos; entre los predictores solo la vivienda va en logaritmo
(asimetría de 1,58 a −0,54), decidido con los datos de entrenamiento.

**11. Horizonte.** Un mes: el mes t se pronostica con los datos de t−1; los 6 meses de cada fold son
bloques de evaluación, no un horizonte de 6 meses. El supuesto es optimista para importaciones y
licencias, que se publican 45 a 50 días después del cierre del mes, y lo declaramos como limitación.

**12. Frente al 4,89 %.** El 4,89 % se calculó sobre los mismos meses con que el artículo ajustó su
modelo, así que solo se compara con nuestro error de ajuste: 12,8 % con el mismo stepwise (11,4 %
sin la pandemia). Compararlo con el de validación mezclaría ajustar con predecir; en meses no vistos
el stepwise da 15,0 %, peor que la media.

**13. La prueba.** La rúbrica de la Segunda entrega (7.1 a 7.4) no la pide; la pide la entrega final
(8.1). Se abre una sola vez y después de elegir el modelo: si la miráramos ahora, cualquier ajuste
posterior la volvería optimista, porque la validación cruzada ya se usó para seleccionar.

**14. Ridge y no KNN.** En el central empatan: KNN da 8,99 % y Ridge 9,08 %, una diferencia de
−0,09 puntos con error estándar de 0,87 en la comparación fold a fold. Desempatamos por estabilidad
(Ridge da 9,02–9,14 % en los tres escenarios; KNN llega a 10,26 % en el alto) e interpretabilidad,
que pide la 7.4; KNN sigue como competidor y se evalúa en la prueba de la entrega final.

**15. Ridge y no Lasso.** También empatan: Lasso da 9,40 % y Ridge 9,08 % (+0,32 ± 0,50), y Ridge le
gana en 3 de 5 folds en los tres escenarios; Lasso es más consistente frente a la media (4 de 5 folds
frente a 3 de 5). Elegimos Ridge porque con predictores correlacionados (inflación y tasa de consumo,
0,85) la selección de Lasso cambia según el periodo (vivienda entra en 3 de 5 folds), mientras Ridge
conserva las 6 variables con signos estables, que es lo que necesitamos para interpretar.

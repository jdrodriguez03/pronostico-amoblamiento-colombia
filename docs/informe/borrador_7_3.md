# 7.3 Entrenamiento y comparación de modelos

Comparamos cinco modelos vistos en el curso. El primero es la regresión lineal múltiple (MLR) con
selección stepwise, que reproduce el método del artículo. A su lado pusimos la misma regresión con
todos los predictores, Ridge y Lasso, que encogen los coeficientes cuando los predictores están
correlacionados (Lasso, además, puede llevarlos a cero), y KNN para regresión, que no supone una
relación lineal y sirve de contraste. Todos modelan el logaritmo del índice de ventas y pasan por el
mismo preprocesamiento. Como referencia mínima usamos dos líneas base: la media histórica de cada
serie y el ingenuo estacional, que repite el valor de doce meses antes.

Para elegir los hiperparámetros probamos una rejilla de valores con validación cruzada temporal: cinco
particiones por mes, cada una entrenada con todo el pasado disponible y validada en los seis meses
siguientes, entre febrero de 2019 y julio de 2025. El criterio fue el MAPE de la serie nacional de
muebles. Todo lo que aprende de los datos, incluida la selección stepwise, se ajustó solo con los
meses de entrenamiento de cada partición. Las cifras que siguen son del escenario central.

Tabla 1. Comparación de modelos por validación cruzada (escenario central)

| Modelo | Hiperparámetro elegido | MAPE nacional (media ± desv.) | Mejora frente a la media | Folds en que gana a la media |
| --- | --- | --- | --- | --- |
| KNN | k = 3, pesos uniformes | 9,0 % ± 4,5 | 12,2 % | 3 de 5 |
| Ridge (modelo preliminar) | alpha = 1000 | 9,1 % ± 6,0 | 11,5 % | 3 de 5 |
| Lasso | alpha ≈ 0,032 | 9,4 % ± 5,7 | 8,3 % | 4 de 5 |
| Media (línea base) | — | 10,3 % ± 5,9 | — | — |
| Ingenuo estacional (línea base) | — | 12,3 % ± 7,8 | −19,9 % | 2 de 5 |
| MLR completa | — | 14,8 % ± 6,5 | −44,3 % | 1 de 5 |
| MLR con stepwise (artículo) | α = 0,25, fijo | 15,0 % ± 6,7 | −46,0 % | 1 de 5 |

En las curvas de validación (Figura 1) se ve que los valores elegidos son mínimos reales. Al principio
el mejor alpha de Ridge y el mejor k de KNN caían en el borde de la rejilla, así que la ampliamos;
con más valores, las dos curvas vuelven a subir. Ningún valor más sencillo resultó igual de bueno al
comparar partición por partición. KNN con un solo vecino ilustra bien el sobreajuste: su error en
entrenamiento es 0 % porque memoriza, y en validación pierde.

Figura 1. Curvas de validación de Ridge, Lasso y KNN: error de validación con su banda de una
desviación, error de entrenamiento y valor elegido.

Los tres mejores modelos superan a las dos líneas base en promedio, aunque la ventaja frente a la
media es modesta, entre 8 % y 12 %, y no se da en todas las particiones (Figura 2). Esperábamos que
el ingenuo estacional fuera la referencia más difícil de superar, y resultó serlo la media. El
método del artículo, en cambio, queda por debajo de las dos.

Figura 2. MAPE de cada partición por modelo, en los tres escenarios de ruido.

Entre KNN, Ridge y Lasso las diferencias son menores que la variación entre particiones: KNN y Ridge
se separan apenas 0,08 puntos, con un error estándar de 0,88 cuando se comparan partición por
partición. Con un empate así, elegimos el modelo preliminar por estabilidad e interpretabilidad.
Ridge da entre 9,0 % y 9,1 % en los tres escenarios y sus coeficientes no cambian de signo de una
partición a otra; KNN, en cambio, sube a 10,2 % en el escenario de ruido alto, apenas 0,03 puntos por debajo de la media, y
no tiene coeficientes que interpretar. Lasso le gana a la media en más particiones, pero con
predictores correlacionados la variable que conserva cambia según el periodo. Fuera de KNN, el orden
de los modelos casi no se mueve entre escenarios.

Tabla 2. El artículo frente a este trabajo

| | Artículo base | Este trabajo |
| --- | --- | --- |
| Error de ajuste (mismos meses con que se estima el modelo) | 4,89 % | 12,8 % con el mismo método; 11,4 % sin abril a agosto de 2020 |
| Error en meses no vistos | No lo reporta | 15,0 % con el mismo método; 9,1 % con Ridge |
| VIF máximo | 23,18 | 6,28 |
| Coeficientes con signo contrario a la teoría | Población | Stepwise: ICC, inflación y vivienda. Ridge: solo inflación |

El 4,89 % del artículo se calculó sobre los mismos meses con que se estimó el modelo, así que solo
se puede comparar con nuestro error de ajuste. El nuestro es mayor por dos razones: incluye la caída
de la pandemia, y un mismo modelo se ajusta a las 56 series del panel en lugar de a una sola serie
desestacionalizada. Lo que el artículo no mide es cuánto se equivoca en meses que no vio, y ahí su
método pierde incluso contra el promedio de cada serie. Usamos el 4,89 % de su tabla de resultados;
las cifras de su resumen (3,47 % y 4,21 %) no coinciden con ella.

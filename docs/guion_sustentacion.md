# Guion de la sustentación

Dueño: Juanes

Las 8 preguntas de la profesora, cada una con una respuesta de máximo dos frases y una cifra, para repasar el miércoles a primera hora. Las cifras salen de `results/tablas/` (corridas del 10 de noviembre). Donde dice *confirmar*, el dueño de esa parte ajusta la frase antes del repaso.

1. **¿Quién usaría el pronóstico?** El área de planeación comercial y de compras de un fabricante o distribuidor de muebles, para anticipar la demanda nacional del mes siguiente. Recibe un índice (2019 = 100) con su rango p10–p90, no unidades ni pesos. *(Confirmar con Julián, 8.2.)*

2. **¿Qué decisión tomaría?** Ajustar al alza o a la baja el plan de compras y producción del mes siguiente según la dirección del índice. En la prueba conviene corregir hacia arriba: el valor observado superó en promedio en un 26 % al pronóstico de KNN. *(Confirmar con Julián, 8.2.)*

3. **¿Cuánto mejora frente a no tener modelo?** Frente a la media de cada serie, KNN reduce el MAPE de prueba un 28 % (27,6 % → 19,8 %). Frente al ingenuo estacional no mejora: queda 3,7 puntos por encima, con un IC por bloques de −0,3 a 7,5 que cruza el cero.

4. **¿Dónde se equivoca más?** Subestima los 12 meses de prueba, con los mayores errores en enero de 2026 (30 %) y noviembre de 2025 (28 %). Por línea, el error se concentra en sonido y video (37 %) y en informática (36 %); en muebles es de 19 %.

5. **¿Qué tan confiable es la parte regional?** Es menos confiable que la nacional: la serie departamental es simulada y solo se cumplen 3 de las 5 verificaciones (fallan la correlación entre líneas y las participaciones departamentales). En la prueba, el MAPE del panel de KNN es 22,5 % y varía de 20,5 % (Atlántico) a 25,7 % (Valle del Cauca), cifras válidas solo dentro de los escenarios. *(Confirmar con Aleja, 8.3–8.4.)*

6. **¿Qué información debe estar disponible al pronosticar?** El primer día hábil del mes t: tasa de consumo de t−1, ICC e inflación de t−2, e importaciones y licencias de vivienda de t−3. Usar solo eso cuesta 0,23 puntos de MAPE de validación en Ridge y 1,01 en KNN.

7. **¿Ridge o KNN, y por qué?** KNN, porque cumple la regla escrita antes de abrir la prueba: 19,8 % frente a 25,7 %, una diferencia mayor que la desviación de Ridge entre folds (5,5), y menor error en 8 de 12 meses. Las dos condiciones se cumplen por poco y el IC de la diferencia (−1,4 a 12,0) cruza el cero, así que la elección es por la regla y no por una superioridad demostrada.

8. **¿Implementar hoy, como piloto o no desplegar?** *Se decide en la llamada de las 21:00.* El criterio acordado para un piloto (que el modelo supere al ingenuo estacional en prueba con un IC que no cruce el cero) no se cumple: KNN queda 3,7 puntos peor que el ingenuo (IC −0,3 a 7,5). Sobre esa base no se recomienda desplegarlo como sustituto del ingenuo estacional.

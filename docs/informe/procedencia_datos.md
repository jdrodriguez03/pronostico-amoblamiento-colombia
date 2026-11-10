# Procedencia de los datos y limitación de la simulación


El índice de cada departamento d, línea l y mes t se construye como **Índice(d, l, t) = L(l, t) × F(d, t) × η(d, l, t)**.

| Componente | De dónde sale | Tipo | Qué conclusiones se pueden sostener con él |
| --- | --- | --- | --- |
| Serie nacional de muebles, las otras 7 líneas nacionales (L) y los 5 predictores (ICC, inflación, tasa de consumo, importaciones cap. 94, área licenciada de vivienda) | DANE (EMC, IPC, importaciones, ELIC), Fedesarrollo y Superintendencia Financiera | **Observado** | El MAPE **nacional** de la línea 8 y la comparación con el artículo: la serie nacional es real y no depende de η. |
| Índices regionales por departamento (F) y pesos base 2019 | DANE, EMC territorial (grupo de comercio del hogar por departamento) | **Observado** | Diferencias generales de nivel y tendencia entre departamentos, sin distinguir por línea. |
| Panel departamento × línea antes del ruido (L × F, ponderado con los pesos) | Combinación de los dos anteriores | **Construido a partir de información real** | Que cada serie regional sigue la tendencia y estacionalidad real de su línea y de su departamento; **no** que sea la serie que el DANE mediría por departamento y línea (no existe publicada). |
| Componente propio η de cada departamento × línea | AR(1) con φ = 0,6 y σ = 2 %, 7 % y 13 % (escenarios bajo, central y alto), semilla 42, reescalado para conservar la serie nacional | **Simulado** | Nada por sí solo. El MAPE del panel (56 series), los resultados por departamento y cualquier lectura regional **solo valen dentro de los escenarios**. |

**Qué no depende de lo simulado.** Como η se reescala para que el agregado nacional reproduzca exactamente la serie real (error máximo 2,8 × 10⁻¹⁴ puntos), el MAPE nacional se calcula contra datos observados. Aun así, los modelos se entrenan sobre el panel, por lo que el MAPE nacional cambia un poco entre escenarios (Ridge 9,02 %–9,14 %).

## Verificación: 3 de 5 pruebas se cumplen

Se cumplen la magnitud del ruido (6,66 % frente al 7 % objetivo en el central), la persistencia (autocorrelación media 0,56 frente a 0,60) y la conservación de la serie nacional. Las dos que no se cumplen:

**Correlación entre líneas.** Se esperaba que los cambios mensuales de dos líneas de un mismo departamento tuvieran una correlación media dentro de la referencia descriptiva de 0,07 a 0,30. En el escenario central salió **0,59** (0,65 en el bajo y 0,50 en el alto); solo 7 de los 196 pares quedan dentro de la referencia. Ocurre porque todas las líneas de un departamento comparten el mismo factor regional F(d, t), las series nacionales de las líneas ya se mueven juntas y η es pequeño frente a esos choques comunes ya que a menor ruido, más correlación, lo que implica que el panel simulado tiene líneas más parecidas entre sí que lo que sugiere la referencia, así que un modelo puede aprovechar esa similitud y verse mejor en el panel de lo que se vería con datos regionales reales. La referencia tampoco se puede validar, porque el DANE no publica ventas por departamento y línea.

**Participaciones departamentales.** Se esperaba que la participación promedio de cada departamento en la serie nacional de muebles coincidiera con su peso base 2019 (tolerancia numérica 10⁻⁶), y según lo observado, no coincide en el central, Bogotá queda 1,2 puntos porcentuales por encima de su peso (23,8 % frente a 22,6 %), Valle del Cauca 1,3 por debajo (10,8 % frente a 12,1 %), Cundinamarca 0,7 por encima y Atlántico 0,5 por debajo. Esto ocurre porque el factor regional real F(d, t) cambia en el tiempo frente a 2019 y η agrega variación propia; los pesos son coeficientes de agregación fijos, no participaciones que deban mantenerse. Implica que el reparto entre departamentos del panel no es exactamente el del año base y que las cifras por departamento llevan ese sesgo.

**Esta es una limitación central del proyecto, no un detalle**: la simulación no está completamente validada, y todo lo regional se presenta como válido solo dentro de los escenarios.

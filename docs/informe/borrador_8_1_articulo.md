# 8.1 Contraste con el artículo (borrador)

Dueño: Julián · Se pega en la 8.1, después del texto de Juanes · Cifras de `comparacion_cv.csv`, `tabla_final.csv` y `vif_final.csv` (escenario central).

| | Artículo | Este trabajo |
| --- | --- | --- |
| Modelo | MLR con stepwise | KNN (final); Ridge y MLR con stepwise como contraste |
| Predictores | Del mismo mes | Solo lo publicado al emitir (rezagos de 1 a 3 meses) |
| MAPE de ajuste (meses de entrenamiento) | 4,89 % | KNN 10,45 %; Ridge 11,31 %; MLR con stepwise 11,31 % |
| MAPE fuera de muestra | — | Validación: KNN 9,77 %; prueba: KNN 19,80 % |
| VIF máximo | 16,2 y 23,2 | 8,7 |

**Qué se reprodujo.** El objetivo del artículo, pronosticar las ventas de muebles con indicadores macroeconómicos y un modelo lineal, se reprodujo con datos colombianos: la MLR con stepwise se reconstruyó dentro del Pipeline y la regularización resolvió la multicolinealidad que el artículo reportaba (VIF máximo 8,7 y signos estables entre folds).

**Qué cambió.** El MAPE de ajuste es más del doble que el del artículo (11,3 % frente a 4,89 % con el mismo modelo), por tres razones: aquí el modelo pronostica 56 series departamento × línea y no una sola serie nacional; solo usa información ya publicada; y el periodo incluye la pandemia y el crecimiento de 2024–2026. Además, el 4,89 % del artículo es error de ajuste; la cifra comparable para un pronóstico real es la de prueba (19,80 %), y en ella ningún modelo supera al ingenuo estacional (16,06 %). La reproducción es, por tanto, parcial: el método se reproduce, pero la precisión del artículo no se sostiene fuera de muestra con información disponible a tiempo.

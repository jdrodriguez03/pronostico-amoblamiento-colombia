# 8.2 Interpretación de resultados (borrador)

Dueño: Julián · Máximo una página en el informe · Cifras de `results/tablas/coeficientes_final.csv`, `vif_final.csv`, `knn_vecinos.csv`, `prueba.csv` y `tabla_final.csv` (`scripts/interpretar_final.py`, solo con entrenamiento).

**Qué pronostica y para quién.** El modelo pronostica el índice de ventas reales de muebles (2019 = 100) del mes siguiente, con lo publicado el primer día hábil de ese mes. Lo usaría un fabricante o comercializador de muebles, o un gremio, en planeación de demanda: una señal de la dirección y la magnitud del mes que empieza para ajustar compras, producción y personal. No dice cuántas unidades comprar: para eso hacen falta unidades, costos, inventario actual y tiempos de reposición.

**Cómo predice KNN, el modelo final.** KNN no tiene coeficientes: pronostica cada departamento × línea como el promedio de los 3 registros de entrenamiento más parecidos. En la práctica, los 3 vecinos son siempre del mismo mes calendario (100 %) y casi nunca de la misma serie (entre 2 % y 34 % según el año): el modelo busca meses del año con condiciones económicas parecidas. Para los registros de 2025, los vecinos vienen en promedio de finales de 2023. Eso explica su principal error: **KNN solo promedia valores ya observados y no puede pronosticar por encima de ellos**, así que cuando el mercado crece se queda corto (8.1 y 8.4).

**Qué variables pesan (lectura con Ridge).** Para leer las variables se usa Ridge, el modelo interpretable de contraste (α = 316), ajustado con los mismos rezagos. Los coeficientes estandarizados son asociaciones condicionadas: el cambio esperado en el logaritmo del índice cuando un predictor sube una desviación estándar y los demás se mantienen fijos, no un efecto causal. Los seis conservan el signo en los 5 folds.

| Predictor (rezago) | Coeficiente (media ± desv. entre folds) | Cambio aproximado del índice por 1 desv. | Signo esperado |
| --- | --- | --- | --- |
| covid | −0,171 ± 0,001 | −15,7 % | Negativo ✓ |
| Inflación (t−2) | +0,086 ± 0,005 | +9,0 % | Negativo ✗ |
| Tasa de consumo (t−1) | −0,061 ± 0,003 | −5,9 % | Negativo ✓ |
| Importaciones (t−3) | +0,048 ± 0,005 | +5,0 % | Positivo ✓ |
| ICC (t−2) | −0,039 ± 0,006 | −3,8 % | Positivo ✗ |
| Área de vivienda (t−3) | −0,003 ± 0,002 | −0,3 % | Positivo (≈ 0) |

El crédito y las importaciones se asocian con las ventas en el sentido que da la teoría. **La inflación con signo positivo no demuestra que más inflación cause más ventas**: entre 2021 y 2023 la inflación subió a la vez que el consumo se recuperaba de la pandemia, así que el coeficiente puede recoger esa recuperación o la tendencia del periodo. Lo mismo vale para el ICC con signo negativo: su información se cruza con la de la inflación y la tasa. El VIF máximo es 8,7 (inflación), frente a 16,2 y 23,2 en el artículo: con los rezagos realistas y la regularización, los coeficientes son estables entre folds, pero siguen sin ser causales. Los p-valores del stepwise no se usan como prueba de capacidad predictiva.

**Tipos de error e implicaciones.** El error dominante es la subestimación: en los 12 meses de prueba lo observado supera a lo pronosticado por KNN en 26 % en promedio, con los mayores errores en noviembre de 2025 (temporada alta) y enero de 2026. Para un usuario, eso significa planear por debajo de la demanda: riesgo de quiebres de inventario más que de sobrestock. La lectura por departamento hereda el componente simulado y vale solo dentro de los escenarios; la serie nacional es real.

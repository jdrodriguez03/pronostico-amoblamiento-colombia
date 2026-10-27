# 7.2 Pipeline de preprocesamiento y modelado (borrador)

Dueño: Juanes · Máximo media página en el informe · Código: `src/modelos/pipeline.py`
(`construir_pipeline`) y `src/modelos/preprocesamiento.py` (`crear_preprocesador`).

**Qué se pronostica.** El índice de ventas del mes t con la información del mes t−1: los cinco
predictores económicos entran con un mes de rezago (`_rez1`), así que es un **pronóstico a un mes**.
Los 6 meses de cada fold de validación son bloques de evaluación, no un horizonte de 6 meses: cada
mes validado usa el dato real del mes anterior. El modelo se ajusta sobre el panel (departamento ×
línea × mes) y se evalúa en la serie nacional de muebles, que se obtiene agregando la línea 8 con
los pesos departamentales fijos de 2019 (`agregar_nacional`).

**Etapas, en orden.** Todo lo que aprende de los datos va dentro de un único objeto de scikit-learn.
`GridSearchCV` y la validación cruzada lo ajustan de cero en cada fold, solo con los meses de
entrenamiento de ese fold; los meses de validación solo se transforman y se predicen. Para cada
etapa se indica qué información del futuro se filtraría si se ajustara fuera, con los 78 meses:

1. **Logaritmo del objetivo** (`TransformedTargetRegressor`): el modelo aprende log(y) y la
   predicción vuelve a la escala del índice con exp, que es donde se miden MAPE, MAD y RMSE. No
   estima parámetros, pero estando dentro la transformación y su inversa se aplican siempre igual.
2. **One-hot de mes, departamento y línea** (`OneHotEncoder`, `drop="first"`; KNN sin `drop`):
   aprende la lista de categorías. Fuera, conocería categorías que solo aparecen en validación; con
   `handle_unknown="ignore"` una categoría nueva no rompe la predicción.
3. **Logaritmo y estandarización del área de vivienda**, y **estandarización de los otros cuatro
   predictores** (`StandardScaler`): aprende media y desviación. Fuera, esas medias incluirían los
   meses validados (por ejemplo, el crecimiento de 2025) y cada fila de validación quedaría
   centrada con información de su propio periodo. `covid` pasa sin cambios.
4. **Selección stepwise** (`StepwiseSelector`, α = 0,25, solo en la MLR con stepwise; las dummies
   no se someten a selección): elige variables por p-valor. Es la fuga más grave: elegidas con todos
   los meses, las variables ya "sabrían" cuáles se relacionan con el índice en los meses que luego se
   validan, y el error saldría optimista.
5. **Modelo** (regresión lineal, Ridge, Lasso o KNN): coeficientes o vecinos guardados. Fuera, los
   meses de validación estarían entre los datos de ajuste o entre los vecinos posibles.
6. **Hiperparámetros** (alpha de Ridge y Lasso; k y pesos de KNN): `GridSearchCV` con
   `CVTemporalPorMes` y `scorer_mape_nacional`, que elige por el MAPE de la serie nacional. Elegirlos
   mirando la prueba (agosto de 2025 a julio de 2026) la contaminaría; por eso la prueba no se abre
   en esta entrega.

La agregación nacional se hace después de predecir y no aprende nada: usa pesos fijos. Las
columnas de auditoría de la simulación (`componentes_regionales.csv`, `lineas_nacionales.csv`) nunca
entran como predictores (`config.COLUMNAS_PROHIBIDAS`), y `tests/test_fuga.py` lo comprueba junto con
que ningún archivo salvo `scripts/evaluar_prueba.py` lea la prueba.

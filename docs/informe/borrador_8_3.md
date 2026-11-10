# 8.3 Aplicación de Streamlit (borrador)

> Dueña: Aleja. Media página.

La aplicación (`app/app.py`, lógica en `src/despliegue/predecir.py`) permite consultar el pronóstico del índice de ventas reales (2019 = 100) para un ámbito (nacional agregado o uno de los 7 departamentos), una de las 8 líneas y un mes. **Recibe** los cinco predictores con el rezago realista de cada uno (tasa de consumo de t−1; ICC e inflación de t−2; importaciones y área de vivienda de t−3), prellenados desde la base y editables, junto con el mes, el departamento y la línea, que se conocen de antemano. Para cada predictor muestra el mes al que corresponde el dato.

**Coherencia entre entrenamiento y uso.** La app prepara las filas con la misma función que el entrenamiento (`agregar_rezagos_reales`) y predice con el mismo `Pipeline`: el one-hot, el logaritmo del área de vivienda, la estandarización y el modelo están dentro de él, así que los datos de uso se transforman con las medias, desviaciones y categorías aprendidas en el entrenamiento. El modelo es el final, KNN con k = 3 (`construir_modelo_final`), y se ajusta con febrero 2019 – julio 2025 al abrir la app (tarda segundos; no se sube ningún modelo al repositorio). Para el ámbito nacional se predice cada departamento y se agrega con los pesos base 2019, igual que en la evaluación.

**Devuelve** el índice pronosticado, el valor observado del mismo mes del año anterior, la variación porcentual y el rango p10–p90 del error fuera de muestra de la validación cruzada. Un recuadro fijo recuerda que es un índice (no unidades, pesos ni inventario), que los predictores son observados (pronóstico *ex post*, salvo agosto de 2026, el único mes ex ante) y, si se elige un departamento, que el componente regional es simulado y la cifra vale solo dentro del escenario central.

**Ejecución:** `pip install -r requirements.txt` y `streamlit run app/app.py` desde la raíz del repositorio.

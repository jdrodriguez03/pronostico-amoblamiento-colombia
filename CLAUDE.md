# Instrucciones para asistentes de código (Claude, Copilot, etc.)

Proyecto académico de Machine Learning I. Antes de proponer cambios:

1. **Solo modelos del curso:** KNN, Naive Bayes, regresión lineal y logística, LDA/QDA y
   regularización (Ridge, Lasso). Nada de árboles, ensambles, SVM ni redes neuronales.
2. **Nada aprende fuera del Pipeline.** Escalado, codificación, selección de variables e
   hiperparámetros van dentro de `construir_pipeline()` y se ajustan en cada fold.
3. **Particiones por mes**, con `CVTemporalPorMes`. Nunca `train_test_split` ni `shuffle=True`.
4. **No leer la prueba** (agosto 2025 – julio 2026). `cargar_prueba()` solo en
   `scripts/evaluar_prueba.py`.
5. `L`, `F` y `eta` nunca son predictores (`config.COLUMNAS_PROHIBIDAS`).
6. Rutas y constantes siempre desde `src/config.py`. Semilla: `config.SEMILLA`.
7. Cada archivo tiene un dueño (primera línea `# Dueño: ...`). No edites archivos de otra persona
   ni cambies nombres o parámetros de funciones existentes.
8. Escribe código y comentarios en español, y explica cada decisión: el equipo debe poder
   sustentarla oralmente.

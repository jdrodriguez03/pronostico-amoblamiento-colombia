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

## Por qué así

(Juanes: ventana creciente, partición por mes, ingenuo estacional con t−12 siempre en entrenamiento.)

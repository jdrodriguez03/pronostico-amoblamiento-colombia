# Cuadre de cifras del informe

Dueño: Juanes

Cada cifra del informe con su sección, el archivo de donde sale y la fila. **Fila** = número de
línea del archivo abierto como texto (la línea 1 es el encabezado), para encontrarla con Ctrl+G en
VS Code. **Clave** = las columnas que identifican la fila, por si la línea se mueve al regenerar.
Las cifras de resultados salen de `python scripts/entrenar.py --escenario todos` (27 de octubre);
si se vuelve a correr y cambia una tabla, este cuadre se revisa.

Estado: ✓ coincide con la tabla · ⚠ hay que corregir el borrador (ver la nota).

Para comprobar una fila sin abrir el archivo:

```python
import pandas as pd
c = pd.read_csv("results/tablas/comparacion_cv.csv")
c.query("escenario == 'central' and modelo == 'knn'")[["mape_media", "mape_desv"]]
```

## 7.1 Datos y verificación de la simulación (borrador de Aleja)

| Cifra en el texto | Archivo | Fila | Clave | Estado |
| --- | --- | --- | --- | --- |
| 5.096 registros; 7 departamentos, 8 líneas, 91 meses | `src/config.py` | — | `DEPARTAMENTOS`, `LINEAS`, `N_MESES` | ✓ |
| σ = 2 %, 7 %, 13 %; semilla 42; ρ = 0,6 | `src/config.py` | — | `ESCENARIOS`, `SEMILLA`, `RHO` | ✓ |
| Magnitud de log(η) dentro de ±20 % de σ (0,0190; 0,0666; 0,1237) | `verificacion_simulacion.csv` | 2–4 | Magnitud de log(eta), por escenario | ✓ |
| Persistencia media "≈ 0,562" | `verificacion_simulacion.csv` | 5–7 | Persistencia AR(1) | ⚠ 0,5630 (bajo), 0,5626 (central), 0,5620 (alto): redondeado a 3 decimales el central es 0,563. Escribir "≈ 0,56" o "entre 0,562 y 0,563" |
| Series fuera de 0,50–0,66 (rango 0,395–0,755 en central) | `verificacion_simulacion.csv` | 6 | Persistencia AR(1), central, `detalle` | ✓ |
| Error máximo de conservación 2,84 × 10⁻¹⁴ | `verificacion_simulacion.csv` | 8–10 | Conservación nacional muebles | ✓ |
| Correlación entre líneas 0,646; 0,594; 0,496 | `verificacion_simulacion.csv` | 11–13 | Correlación de cambios entre líneas | ✓ |
| 21 comparaciones de participaciones; diferencia máxima 0,0136188 | `verificacion_simulacion.csv` | 14–34 | Proporción departamental; máx. en Bogotá, bajo (fila 16) | ✓ |

## 7.2 Pipeline (borrador de Juanes)

Solo tiene constantes del diseño, que salen del código y no de una tabla.

| Cifra en el texto | Archivo | Clave | Estado |
| --- | --- | --- | --- |
| Pronóstico a un mes (predictores `_rez1`) | `src/config.py` | `USAR_REZAGO = True` | ✓ |
| Bloques de 6 meses, 5 folds | `src/config.py` | `CV_MESES_VALIDACION`, `CV_FOLDS` (los usa `CVTemporalPorMes`) | ✓ |
| Pesos departamentales de 2019, línea 8 | `src/config.py` | `PESOS_DEPTO`, `LINEA_OBJETIVO_CODIGO` | ✓ |
| Stepwise α = 0,25 | `src/config.py` | `ALFA_STEPWISE` | ✓ |
| Prueba agosto de 2025 – julio de 2026 | `src/config.py` | `INICIO_PRUEBA` | ✓ |

## 7.3 Entrenamiento y comparación (borrador de Julián)

| Cifra en el texto | Archivo | Fila | Clave | Estado |
| --- | --- | --- | --- | --- |
| Rejilla Ridge: 17 valores de 0,001 a 100.000; Lasso: 11 de 0,0001 a 10; KNN k en {1,…,40} | `src/modelos/candidatos.py` | — | `CANDIDATOS` | ✓ |
| Ridge alpha = 1000, 9,1 %; Lasso alpha ≈ 0,032, 9,4 %; KNN k = 3, 9,0 % | `comparacion_cv.csv` | 16–18 | central, knn / ridge / lasso | ✓ |
| Ridge con alpha = 3162: 9,6 % | `curvas_validacion.csv` | 15 | ridge, alpha = 3162,28 | ✓ |
| MLR con stepwise 15,0 % ± 6,7 | `comparacion_cv.csv` | 22 | central, mlr_stepwise | ✓ |
| KNN 9,0 % ± 4,5; Ridge 9,1 ± 6,0; Lasso 9,4 ± 5,7; media 10,3 ± 5,9; ingenuo 12,3 ± 7,8; MLR completa 14,8 ± 6,5 | `comparacion_cv.csv` | 16–22 | central | ✓ |
| Mejoras vs. ingenuo 26,8 / 26,2 / 23,6 / 16,6 / −20,3 / −21,8 % | `comparacion_cv.csv` | 16–22 | central, `mejora_vs_ingenuo` | ✓ |
| Mejoras vs. media 12,2 / 11,5 / 8,3 / −19,9 / −44,3 / −46,0 % | `comparacion_cv.csv` | 16–22 | central, `mejora_vs_media` | ✓ |
| Folds en que gana a la media: KNN 3, Ridge 3, Lasso 4, ingenuo 2, MLR completa 1, stepwise 1 | `comparacion_pareada.csv` | 44–49 | central, referencia = media | ✓ (desde hoy en el CSV) |
| MAPE de ajuste: stepwise 12,8 %, Ridge 13,0 %; sin abril–agosto de 2020: 11,4 % y 11,3 % | `comparacion_articulo.csv` | 2 y 4 | mlr_stepwise, ridge | ✓ |
| VIF máximo 6,28 (inflación) | `vif_inicial.csv` | 3 | inflacion_anual_rez1 | ✓ |
| Stepwise retiene las 6 con p < 1e-4 | `stepwise_seleccion.csv` | 2–7 | las 6 filas `entra` | ✓ (el mayor p es 1,03 × 10⁻⁴, inflación: "p < 1e-4" queda justo en el límite; mejor "p ≤ 1,03 × 10⁻⁴" o "p < 0,001") |
| Lasso anula ICC, tasa de consumo y covid | `estabilidad_coeficientes.csv` | 15, 17, 19 | lasso, `folds_distinto_de_cero = 0` | ✓ |
| [pendiente en el borrador] ¿cambia el orden entre escenarios? | `rango_escenarios.csv` | 2–8 | columnas `puesto_*` | KNN pasa del 1.º al 3.º en alto; Ridge, del 2.º al 1.º |

## 7.4 Validación, métricas y variabilidad (borrador de Juanes)

| Sección | Modelo | Cifra en el texto | Archivo | Fila | Clave |
| --- | --- | --- | --- | --- | --- |
| Tabla | KNN | MAPE 9,00 ± 4,54; MAD 11,92; RMSE 14,74; mejora vs media 12,2 %; vs ingenuo 26,8 %; k = 3, uniformes | `comparacion_cv.csv` | 16 | central, knn |
| Tabla | KNN | gana al ingenuo en 4 de 5 | `comparacion_pareada.csv` | 38 | central, knn, ref. ingenuo |
| Tabla y variabilidad | KNN | escenarios 9,00 – 10,22; rango 1,22; puestos 1/1/3 | `rango_escenarios.csv` | 2 | knn |
| Tabla | Ridge | MAPE 9,08 ± 6,04; MAD 12,42; RMSE 14,23; mejora vs media 11,5 %; vs ingenuo 26,2 %; alpha = 1000 | `comparacion_cv.csv` | 17 | central, ridge |
| Tabla | Ridge | gana al ingenuo en 2 de 5 | `comparacion_pareada.csv` | 39 | central, ridge, ref. ingenuo |
| Tabla y variabilidad | Ridge | escenarios 9,02 – 9,14; rango 0,12; puestos 2/2/1 | `rango_escenarios.csv` | 3 | ridge |
| Tabla | Lasso | MAPE 9,40 ± 5,66; MAD 12,88; RMSE 15,28; mejora vs media 8,3 %; vs ingenuo 23,6 %; alpha = 0,03162 | `comparacion_cv.csv` | 18 | central, lasso |
| Tabla | Lasso | gana al ingenuo en 4 de 5 | `comparacion_pareada.csv` | 40 | central, lasso, ref. ingenuo |
| Tabla y variabilidad | Lasso | escenarios 9,30 – 9,47; rango 0,17; puestos 3/3/2 | `rango_escenarios.csv` | 4 | lasso |
| Tabla | Media | MAPE 10,26 ± 5,86; MAD 13,94; RMSE 15,82; mejora vs ingenuo 16,6 % | `comparacion_cv.csv` | 19 | central, media |
| Tabla | Media | gana al ingenuo en 3 de 5 | `comparacion_pareada.csv` | 41 | central, media, ref. ingenuo |
| Tabla y variabilidad | Media | escenarios 10,26 – 10,26; rango 0,00 | `rango_escenarios.csv` | 5 | media |
| Tabla | Ingenuo | MAPE 12,30 ± 7,76; MAD 15,86; RMSE 18,54; mejora vs media −19,9 % | `comparacion_cv.csv` | 20 | central, ingenuo_estacional |
| Tabla y variabilidad | Ingenuo | escenarios 12,30 – 12,30; rango 0,00 | `rango_escenarios.csv` | 6 | ingenuo_estacional |
| Tabla | MLR completa | MAPE 14,80 ± 6,52; MAD 18,83; RMSE 22,11; mejora vs media −44,3 %; vs ingenuo −20,3 % | `comparacion_cv.csv` | 21 | central, mlr_completa |
| Tabla | MLR completa | gana al ingenuo en 1 de 5 | `comparacion_pareada.csv` | 42 | central, mlr_completa, ref. ingenuo |
| Tabla y variabilidad | MLR completa | escenarios 14,49 – 14,80; rango 0,31 | `rango_escenarios.csv` | 7 | mlr_completa |
| Tabla | MLR stepwise | MAPE 14,98 ± 6,73; MAD 19,04; RMSE 22,31; mejora vs media −46,0 %; vs ingenuo −21,8 % | `comparacion_cv.csv` | 22 | central, mlr_stepwise |
| Tabla | MLR stepwise | gana al ingenuo en 1 de 5 | `comparacion_pareada.csv` | 43 | central, mlr_stepwise, ref. ingenuo |
| Tabla y variabilidad | MLR stepwise | escenarios 14,53 – 14,98; rango 0,45 | `rango_escenarios.csv` | 8 | mlr_stepwise |
| Variabilidad | Todos | desviación entre folds de 4,5 a 7,8 | `comparacion_cv.csv` | 16 y 20 | central, `mape_desv` mínimo (knn) y máximo (ingenuo) |
| Variabilidad | KNN − Lasso | distancia entre los 3 mejores: 0,40 puntos | `comparacion_cv.csv` | 16 y 18 | central, `mape_media` de knn y lasso |
| Variabilidad | KNN | fold 5: 16,47 % (el menor) | `cv_resultados.csv` | 71 | central, knn, fold 5 |
| Variabilidad | MLR stepwise / completa | fold 5: 21,11 % (el mayor) | `cv_resultados.csv` | 51 y 56 | central, fold 5 |
| Variabilidad | Ingenuo | fold 1: 21,32 % | `cv_resultados.csv` | 42 | central, ingenuo_estacional, fold 1 |
| Variabilidad | Todos | RMSE entre 1,13 y 1,24 veces el MAD | `comparacion_cv.csv` | 19 y 16 | central, `rmse / mad` de media (1,135) y knn (1,237) |
| Comparación por fold | KNN, Ridge, Lasso, media, ingenuo | ganan a la stepwise en 4 de 5 | `comparacion_pareada.csv` | 50–54 | central, ref. mlr_stepwise |
| Comparación por fold | KNN / Ridge / Lasso | ganan a la media en 3 / 3 / 4 de 5; diferencia media −1,25 / −1,18 / −0,86 | `comparacion_pareada.csv` | 44–46 | central, ref. media |
| Limitaciones | KNN y media (alto) | 10,22 frente a 10,26: 0,03 puntos | `comparacion_cv.csv` | 4 y 5 | alto, knn y media |
| Limitaciones | KNN | el desempate cambia el MAPE en centésimas a décimas | `docs/decisiones.md` | entrada del 26-oct de Juanes | no es una cifra de resultados: viene de la investigación de los empates |
| Limitaciones | — | importaciones y ELIC salen 45 a 50 días después del mes | `data/raw/FUENTES.md` y `docs/decisiones.md` | entrada del 25-oct de Julián | fechas de publicación |

## 7.4 Interpretación (borrador de Julián)

Pendiente: se agrega cuando exista `docs/informe/borrador_7_4_interpretacion.md` y
`results/tablas/coeficientes.csv`.

## Cifras del artículo (no salen de nuestras tablas)

| Cifra | Fuente |
| --- | --- |
| MAPE 4,89 %, MAD 447,14, MSD 417.422,39; Holt–Winters 6 % | Artículo base, tabla de resultados (`docs/decisiones.md`, 2026-10-09) |
| VIF 23,18 (precios al productor) y 16,16 (población) | Artículo base (`docs/decisiones.md`, Entrega 1) |
| R² 76,73 % (ajustado 74,09 %); población −0,265 | Artículo base (citado en `borrador_7_3.md`) |

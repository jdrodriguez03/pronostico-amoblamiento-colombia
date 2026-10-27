# Declaración del uso de IA generativa

Dueño: Julián

Para qué se usó, en qué partes y cómo el equipo verificó y entiende cada decisión. La rúbrica
permite la IA generativa como apoyo "siempre que se declare y que el equipo comprenda y pueda
sustentar cada decisión técnica".

## Herramienta

Claude (Anthropic), como asistente de conversación y como asistente de código. Los commits que
el asistente preparó directamente aparecen en el historial del repositorio con autor `Claude`.

## Para qué se usó

| Parte del proyecto | Uso de la IA | Qué hizo el equipo |
| --- | --- | --- |
| Construcción de la base (Entrega 1) | Formular la base de datos: combinar las series reales del DANE con un componente simulado, el planteamiento L × F × η, los tres escenarios de ruido y las pruebas de verificación | Elegir las fuentes, decidir la estructura de la simulación y comprobar que la base reproduce exactamente la serie nacional real (`results/verificacion.md`) |
| Organización | Proponer la guía día por día, el plan de cierre y la división de tareas | Ajustar el plan a la disponibilidad real y decidir qué recortar |
| Revisión metodológica | Contrastar el repositorio con el contenido del curso y señalar riesgos: comparación con el 4,89 % del artículo, horizonte de pronóstico, error de entrenamiento, bordes de rejilla | Decidir qué se corrige y qué se declara como limitación (`docs/decisiones.md`) |
| Código | Generar código en general (construcción de la base, predictores, modelos, evaluación, pruebas y notebooks) y revisarlo | Leer cada cambio, correr `pytest` y el notebook, y aplicarlo con un PR revisado por otro integrante |
| Diagnóstico | Detectar problemas al correr los modelos: KNN no reproducible por empates en distancias, mínimos en el borde de las rejillas | Confirmar cada hallazgo con los datos antes de cambiar el código |
| Redacción | Primeros borradores de documentos y del texto del informe | Revisar, corregir y validar cada cifra contra su tabla de `results/tablas/` |

## Cómo se controló

- **Reglas para el asistente.** `CLAUDE.md` le fija las reglas del proyecto: solo modelos del
  curso, nada que aprenda de los datos fuera del Pipeline, particiones por mes, la prueba
  cerrada, rutas desde `config.py` y cada archivo con su dueño.
- **Nada entra sin pruebas.** Todo cambio pasa por `pytest`, `ruff` y el CI, y por un PR que
  revisa otro integrante.
- **Las cifras salen del código.** Ningún número del informe se toma de la conversación con la
  IA: cada uno se rastrea a su CSV en `docs/cuadre_cifras.md`.
- **Comprensión.** Cada integrante debe poder explicar su parte en la sustentación; para eso
  están el banco de preguntas y los ensayos.

## Uso de cada integrante

Cada quien completa su fila antes de la entrega.

| Integrante | Para qué usó IA generativa |
| --- | --- |
| Julián | Formulación de la nueva base de datos de la Entrega 1 (componente simulado y su verificación), generación general de código, revisión metodológica contra el material del curso, planeación del cierre, notebook de curvas y estabilidad (`04_interpretacion.ipynb`), ajustes a los candidatos y explicación de conceptos (regularización, KNN, multicolinealidad, comparación pareada) para preparar la sustentación |
| Juanes | [completar] |
| Aleja | [completar] |

## Párrafo para el informe

> **Uso de inteligencia artificial generativa.** Se usó Claude (Anthropic) como apoyo para
> formular la construcción de la base de datos (la combinación de series reales del DANE con un
> componente simulado), generar y revisar código, organizar el plan de trabajo, revisar la
> metodología frente al material del curso y redactar borradores. Cada cambio de código se
> verificó con pruebas automáticas y la revisión de otro integrante, y cada cifra del informe sale
> de las tablas que genera el código del repositorio. Las decisiones técnicas fueron tomadas por el equipo, que
> las comprende y puede sustentarlas. El detalle está en `docs/uso_ia.md`.

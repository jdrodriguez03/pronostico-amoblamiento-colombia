# Pronóstico Amoblamiento: guía rápida del repositorio

> Versión en el repo de la guía compartida del equipo. Si hay diferencias, manda la versión del documento compartido.

Oct 8, 2026 · @Julian

## La idea en tres líneas

El repositorio `pronostico-amoblamiento-colombia` tiene todo lo de la Entrega 2: los datos crudos, los scripts que construyen la base, el flujo de modelamiento y los resultados que van al informe. Está organizado para que tres personas trabajen a la vez sin pisarse y para que el profesor lo reproduzca con un solo comando.

1. **Cada carpeta hace una sola cosa.** `src/datos/` carga y valida la base, `src/modelos/` define los modelos, `src/evaluacion/` mide. Los notebooks solo analizan y grafican: no definen funciones que otros usen.
2. **Cada archivo tiene un dueño.** Solo el dueño lo edita. Si necesitas algo de un archivo ajeno, se lo pides por el chat. La primera línea de cada archivo dice `# Dueño: ...`.
3. **Nadie sube directo a main.** main es la versión que siempre funciona. Cada quien trabaja en su **rama** y la junta con main mediante un **PR** (pull request: una solicitud para que el equipo revise y apruebe tus cambios) en el cierre de cada bloque.

El repositorio está en `github.com/jdrodriguez03/pronostico-amoblamiento-colombia`, privado, en la cuenta de Julián. Ya tiene la base de la Entrega 1 completa y verificada: los insumos en `data/raw/`, los scripts que la construyen en `src/simulation/` y `src/features/`, y las bases finales en `data/processed/`. También tiene la estructura de la Entrega 2 (configuración, contrato, CI y 20 pruebas en verde). Aleja es la dueña de la configuración (`config.py`, CI, README). El 27 de octubre se invita al profesor o se hace público, según se decida en la reunión de arranque.

### Quién cuida qué

| Persona | Letra | Rol | Archivos suyos |
| --- | --- | --- | --- |
| Aleja | A | Variable objetivo, simulación y verificación (7.1); configuración y auditoría de fuga | `src/simulation/`, `src/config.py`, `src/datos/` (`esquema.py`, `cargar.py`), `src/evaluacion/agregacion.py`, `scripts/reproducir.py`, `results/verificacion.md`, `notebooks/01_verificacion_simulacion.ipynb`, `notebooks/05_escenarios.ipynb`, `tests/test_fuga.py`, `README.md`, `pyproject.toml`, `requirements.txt`, `.github/` |
| Julián | B | Predictores y datos, modelos candidatos e interpretación | `src/features/`, `data/` (salvo `fixtures/`), `src/modelos/stepwise.py`, `src/modelos/candidatos.py`, `src/evaluacion/interpretacion.py`, `notebooks/02_calidad_predictores.ipynb`, `notebooks/04_interpretacion.ipynb`, `docs/diccionario_datos.md`, `docs/uso_ia.md`, `docs/bitacora_resultados.md` |
| Juanes | C | Validación, flujo (Pipeline) y resultados | `src/modelos/preprocesamiento.py`, `pipeline.py`, `lineas_base.py`, `src/evaluacion/particion.py`, `metricas.py`, `validar.py`, `src/figuras.py`, `scripts/crear_fixture.py`, `entrenar.py`, `evaluar_prueba.py`, `data/fixtures/`, `notebooks/03_comparacion_modelos.ipynb`, `docs/validacion.md`, `docs/guion_sustentacion.md`, `docs/cuadre_cifras.md` |

Los scripts de construcción de la base (`src/simulation/` y `src/features/`) y los insumos de `data/raw/` no se modifican sin acuerdo de los tres: reproducen byte a byte la base verificada.

La letra va en el nombre de tu rama (`b/e2-predictores`) para saber de quién es cada cosa de un vistazo. Cada quien escribe las pruebas de sus propios módulos en `tests/test_<modulo>.py`. Hay un solo archivo compartido, `docs/decisiones.md`: ahí cualquiera puede **agregar** entradas al final, pero nadie edita las de otro.

El reparto sigue las secciones de la rúbrica: Aleja responde por la 7.1 (consolidación de datos), Juanes por la 7.2 (flujo sin fuga) y la parte de métricas de la 7.4, y Julián por la 7.3 (comparación de modelos) y la interpretación de la 7.4. Aun así, los tres deben poder sustentar todo: por eso hay tres sesiones de "cruce de conocimiento" en la guía día por día.

## El árbol de carpetas

El repositorio se agrupa en tres bloques: **la base** (`data/`, `src/simulation/`, `src/features/`), **el modelamiento** (`src/datos/`, `src/modelos/`, `src/evaluacion/`, `scripts/`) y **los resultados y documentos** (`results/`, `reports/`, `docs/`, `notebooks/`).

```
pronostico-amoblamiento-colombia/
├─ README.md               qué es, cómo instalar y cómo reproducir (Aleja)
├─ pyproject.toml · requirements.txt · .gitignore · .gitattributes · CLAUDE.md
│
│ ── LA BASE (Entrega 1, verificada) ─────────────────────────────
├─ data/
│  ├─ README.md            diccionario de datos
│  ├─ raw/                 8 insumos originales + FUENTES.md (nadie los edita)
│  ├─ interim/             base_{bajo,central,alto}.csv, predictores_*.csv,
│  │                       componentes_regionales.csv y lineas_nacionales.csv (auditoría)
│  ├─ processed/           base_final_{bajo,central,alto}.csv: lo que entra a los modelos
│  └─ fixtures/            base_mini.csv, recorte para pruebas rápidas (Juanes)
├─ src/
│  ├─ simulation/simular_base.py   variable objetivo y verificación (Aleja)
│  ├─ features/predictores.py      predictores y unión con la base (Julián)
│
│ ── EL MODELAMIENTO (Entrega 2) ─────────────────────────────────
│  ├─ config.py            rutas, columnas, pesos, fechas, semilla (Aleja)
│  ├─ figuras.py           estilo común y guardar_figura() (Juanes)
│  ├─ datos/
│  │  ├─ esquema.py        contrato: COLUMNAS_BASE y validar_base() (Aleja)
│  │  └─ cargar.py         cargar_entrenamiento() y cargar_prueba() (Aleja)
│  ├─ modelos/
│  │  ├─ preprocesamiento.py · pipeline.py · lineas_base.py   (Juanes)
│  │  └─ stepwise.py · candidatos.py                         (Julián)
│  └─ evaluacion/
│     ├─ particion.py · metricas.py · validar.py   (Juanes)
│     ├─ agregacion.py     panel → línea nacional de muebles (Aleja)
│     └─ interpretacion.py VIF, coeficientes, estabilidad (Julián)
├─ scripts/
│  ├─ crear_fixture.py · entrenar.py · evaluar_prueba.py   (Juanes)
│  └─ reproducir.py        corre todo en orden (Aleja)
├─ tests/                  pruebas automáticas, cada dueño las suyas
│
│ ── RESULTADOS Y DOCUMENTOS ─────────────────────────────────────
├─ results/
│  ├─ verificacion.md      las 5 pruebas de la base (la escribe simular_base.py)
│  ├─ tablas/ · figuras/   lo que alimenta el informe
├─ notebooks/              01 a 05, uno por dueño
├─ reports/
│  ├─ entrega_1/           informe corregido y cambios por retroalimentación
│  └─ referencias/         artículo base, rúbrica y catálogo
├─ docs/                   guías, decisiones, validación, borradores del informe
└─ .github/                CI, CODEOWNERS y plantilla de PR
```

No hay carpeta `app/` todavía: el despliegue en Streamlit es de la entrega final y se agrega después del 28.

### Qué va y qué no va en cada carpeta

| Carpeta | Qué va adentro | Qué NO va |
| --- | --- | --- |
| `src/datos/` | Funciones que leen archivos y devuelven DataFrames | Modelos, métricas, gráficas |
| `src/modelos/` | Estimadores y transformadores compatibles con scikit-learn | Lectura de archivos, rutas escritas a mano |
| `src/evaluacion/` | Particiones, métricas, agregación, interpretación | Ajuste de modelos fuera de un Pipeline |
| `scripts/` | Programas cortos que llaman funciones de `src/` y guardan archivos | Lógica nueva: si es reutilizable, va a `src/` |
| `notebooks/` | Exploración, tablas y figuras para el informe, importando de `src/` | Funciones que otros necesiten importar |
| `data/raw/` | Los archivos tal como se descargaron | Ningún archivo editado a mano |
| `data/processed/` | Solo lo que genera un script | Nada creado desde un notebook |
| `results/` | Verificación de la base, tablas y figuras generadas por código | Capturas de pantalla, Excel editados a mano |
| `docs/` | Decisiones, diccionario, borradores, guías | Código |

### Los notebooks, sin conflictos

Git no sabe mezclar dos versiones de un mismo notebook. Por eso cada notebook tiene un solo dueño y nadie abre y guarda el de otro (si solo quieres mirarlo, ábrelo en GitHub). Antes de cada commit de un notebook: **Kernel → Restart & Run All**, para que corra de arriba a abajo sin depender de celdas viejas. La primera celda siempre es `from src.config import ...`, y toda figura que vaya al informe se guarda con `guardar_figura()` en `results/figuras/`.

## De dónde salen los datos

Todo lo que entra a los modelos sale de `data/raw/` mediante dos scripts que ya existen y están verificados: `python src/simulation/simular_base.py` y luego `python src/features/predictores.py`. Con la semilla 42 reproducen byte a byte las bases. Nadie edita a mano un archivo de `data/interim/` ni de `data/processed/`.

```
data/raw/  (8 anexos DANE, Superfinanciera, Fedesarrollo)
   │  src/simulation/simular_base.py
   ▼
data/interim/base_{bajo,central,alto}.csv  +  componentes_regionales.csv, lineas_nacionales.csv (auditoría)
   │  src/features/predictores.py
   ▼
data/processed/base_final_{bajo,central,alto}.csv   (19 columnas, validar_base)
   │  src/datos/cargar.py
   ├─ cargar_entrenamiento()  feb 2019 – jul 2025  → CV temporal por mes → modelos
   └─ cargar_prueba()         ago 2025 – jul 2026  → solo scripts/evaluar_prueba.py (día 12)
   ▼
results/tablas/ y results/figuras/  →  informe
```

La variable objetivo y los predictores se unen en `base_final_{escenario}.csv`. Después se separa por mes en entrenamiento y prueba, y la prueba queda cerrada hasta el día 12.

### El contrato: las 19 columnas de la base

Es el acuerdo entre la base y el modelamiento. `validar_base(df)` en `src/datos/esquema.py` lo revisa (las tres bases reales lo cumplen, y hay pruebas que lo comprueban). Solo se cambia en un cierre de bloque. El diccionario completo está en `data/README.md`.

| Columnas | Qué son | Uso en el modelo |
| --- | --- | --- |
| `fecha`, `anio`, `linea_codigo`, `es_muebles` | Identificación de la fila | No son predictores |
| `mes`, `departamento`, `linea` | Estacionalidad y efectos fijos (7 dominios sin tildes, 8 líneas) | Categóricas (one-hot) |
| `indice_ventas_real` | **Variable objetivo** (2019 = 100); es lo único que cambia entre escenarios | Objetivo, en logaritmo |
| `icc`, `inflacion_anual`, `tasa_consumo`, `importaciones_muebles_usd_miles`, `area_vivienda_m2` | Predictores del mismo mes | Solo si `USAR_REZAGO = False` |
| Los mismos con `_rez1` | Un mes de rezago | Predictores por defecto (`config.predictores_modelo()`) |
| `covid` | 1 de abril a agosto de 2020 | Binaria |

`area_vivienda_m2_rez1` es el único vacío permitido: enero de 2019 (56 filas). Con `USAR_REZAGO = True`, `cargar_entrenamiento` descarta ese mes.

```python
from src.datos.cargar import cargar_entrenamiento

df = cargar_entrenamiento("central")        # base real, feb 2019 a jul 2025
df = cargar_entrenamiento(fixture=True)     # base pequeña para pruebas rápidas
```

### La regla de oro: nada aprende fuera del Pipeline

Esto es lo que califica el criterio "Flujo sin fuga de datos". Son seis reglas y aplican a todo el código:

1. **Se parte por mes, nunca por fila.** Los 56 registros de un mes (7 departamentos × 8 líneas) quedan juntos en entrenamiento o en validación. Nunca `shuffle=True` ni `train_test_split`.
2. **Todo lo que aprende de los datos va dentro del Pipeline**: las categorías del OneHotEncoder, las medias del StandardScaler, las variables que elige el stepwise, el α de Ridge y Lasso y el k de KNN. Así se ajustan solo con los meses de entrenamiento de cada fold.
3. **La prueba está cerrada.** `cargar_prueba()` solo se llama desde `scripts/evaluar_prueba.py`, y ese script se corre una vez, el día 12. Antes, nadie calcula una métrica, una media ni una gráfica con agosto de 2025 a julio de 2026.
4. **Las columnas de auditoría nunca son predictores.** Las de `componentes_regionales.csv` y `lineas_nacionales.csv` están en `config.COLUMNAS_PROHIBIDAS` y `validar_base` falla si aparecen: `L_8` es literalmente la respuesta nacional.
5. **Solo se usa el pasado.** Los modelos usan los predictores `_rez1` (el dato del mes t no se conoce al pronosticarlo); el ingenuo estacional usa t−12.
6. **Los escenarios se fijan antes de modelar.** σ = 2 %, 7 % y 13 %, φ = 0,6 y la semilla viven en `config.py` desde el día 1 y no cambian después de ver resultados.

`tests/test_fuga.py` (Aleja) revisa las reglas 1, 3 y 4 en cada PR. Una limitación que no es fuga pero hay que declarar: en los meses de prueba los predictores son valores observados (pronóstico ex post), igual que en el artículo. Va en `docs/decisiones.md` y en el informe.

## config.py: la hoja de ajustes

Todo número o ruta que usan dos personas vive una sola vez en `src/config.py`, y los demás lo importan: `from src.config import OBJETIVO, SEMILLA`. Su dueña es Aleja; si necesitas cambiar un valor, se lo pides a ella, porque afecta a los tres. Sus valores coinciden con la base real y hay pruebas que lo verifican.

| Grupo | Nombres | Valor |
| --- | --- | --- |
| Rutas | `RUTA_RAW`, `RUTA_INTERIM`, `RUTA_PROCESSED`, `RUTA_TABLAS`, `RUTA_FIGURAS`, `ruta_base(escenario)` | Se arman desde la carpeta del repo |
| Reproducibilidad | `SEMILLA`, `ESCENARIOS`, `RHO` | 42; bajo 2 %, central 7 %, alto 13 %; 0,6 |
| Estructura | `DEPARTAMENTOS`, `PESOS_DEPTO`, `LINEAS`, `LINEA_OBJETIVO_CODIGO` | 7 dominios sin tildes, pesos de `componentes_regionales.csv`, código → nombre de las 8 líneas, 8 |
| Fechas | `FECHA_INICIO`, `FECHA_FIN`, `INICIO_PRUEBA` | 2019-01-01, 2026-07-01, 2025-08-01 |
| Columnas | `OBJETIVO`, `PREDICTORES_NUM`, `PREDICTORES_REZ1`, `PREDICTORES_CAT`, `PREDICTORES_BIN`, `COLUMNAS_PROHIBIDAS` | `indice_ventas_real`, los 5 predictores, sus `_rez1`, mes · departamento · línea, `covid`, columnas de auditoría |
| Decisiones | `USAR_REZAGO`, `LOG_OBJETIVO`, `ALFA_STEPWISE` | True, True, 0.25 |
| Validación | `CV_FOLDS`, `CV_MESES_VALIDACION` | 5, 6 |

## El CI: el semáforo de cada PR

Cada vez que alguien abre un PR o sube a main, GitHub lee `.github/workflows/ci.yml`, alquila un computador vacío, instala Python 3.12 y `requirements.txt`, y corre `ruff check .` (estilo) y `pytest` (pruebas). Si los dos pasan, el PR queda en verde; si uno falla, queda en rojo y no se fusiona.

Como arranca desde cero, atrapa el "en mi máquina funciona". Las pruebas usan el fixture, así que el CI tarda menos de dos minutos. Antes de abrir tu PR corre lo mismo en tu terminal: `ruff check . && pytest`.

## Las 6 reglas

| # | Regla | Qué hacer | Qué NO hacer |
| --- | --- | --- | --- |
| 1 | Nunca trabajes en main | `git checkout -b c/e3-pipeline` antes de editar | Editar con `git branch` mostrando `* main` |
| 2 | Solo edita tus archivos | Juanes necesita otra transformación: se la pide a Julián por el chat | Juanes edita `src/modelos/stepwise.py` "solo un momento" |
| 3 | No cambies nombres ni parámetros de funciones ajenas | Agregar un parámetro opcional con valor por defecto y avisar | Renombrar `cargar_entrenamiento` a `leer_train` |
| 4 | Rutas y constantes siempre desde `config.py` | `from src.config import RUTA_TABLAS` | `pd.read_parquet("C:/Users/.../dataset.parquet")` |
| 5 | Nada aprende fuera del Pipeline | Escalar dentro de `construir_pipeline()` | `StandardScaler().fit(df)` sobre todo el dataset |
| 6 | La prueba no se toca antes del día 12 | Comparar modelos con `evaluar_cv()` | Mirar "solo por curiosidad" el MAPE en 2025–2026 |

**Por qué:** si main se rompe, se rompe para los tres (1). Si dos personas editan el mismo archivo, git no sabe cuál conservar (2). Los demás ya escribieron código que llama a esa función con ese nombre (3). Una ruta escrita a mano solo funciona en tu computador (4). Las reglas 5 y 6 son la diferencia entre "Excelente" y "Deficiente" en el criterio de fuga: un vistazo a la prueba ya contamina la elección del modelo.

## Mensajes de commit

Formato `prefijo: qué hiciste`, en presente, minúscula, sin punto final, máximo unas 70 letras y una idea por commit.

| Prefijo | Úsalo cuando... | Ejemplo |
| --- | --- | --- |
| `datos:` | cambias algo de `src/datos/` | `datos: simulo eta como AR(1) con conservación` |
| `modelos:` | cambias algo de `src/modelos/` | `modelos: agrego StepwiseSelector con alfa 0.25` |
| `eval:` | cambias algo de `src/evaluacion/` | `eval: MAPE sobre la línea nacional agregada` |
| `scripts:` | cambias algo de `scripts/` | `scripts: entrenar acepta --escenario todos` |
| `notebooks:` | cambias un notebook | `notebooks: verifico autocorrelación de eta` |
| `resultados:` | subes tablas o figuras regeneradas | `resultados: evaluación única en prueba` |
| `fix:` | arreglas un error, en cualquier carpeta | `fix: el fold 3 incluía julio de 2025 dos veces` |
| `tests:` | agregas o cambias pruebas | `tests: ningún mes aparece en dos particiones` |
| `docs:` | cambias documentos o el README | `docs: agrego decisión sobre objetivo en log` |
| `chore:` | mantenimiento (CI, .gitignore, librerías) | `chore: fijo versión de scikit-learn` |

No sirven `cambios`, `listo` ni `fix: cosas`: en una semana nadie sabrá qué eran.

## Los comandos del día

Todos se corren desde la raíz del repositorio y, en Windows, en Git Bash.

| Paso | Cuándo | Comando | Qué hace |
| --- | --- | --- | --- |
| 1 | Al empezar el bloque | `git checkout main` y `git pull` | Trae lo que se fusionó en el cierre anterior |
| 2 | Justo después | `git checkout -b b/e2-predictores` | Crea tu rama: tu letra / bloque-tema |
| 3 | Mientras trabajas | `git status` | Muestra qué cambió |
| 4 | Al terminar algo pequeño | `git add src/modelos/stepwise.py` y `git commit -m "datos: ..."` | Guarda ese avance; nunca `git add .` |
| 5 | Al final de cada día | `git push -u origin b/e2-predictores` (después solo `git push`) | Sube tu rama como respaldo |
| 6 | Día de cierre, antes de las 18:00 | En GitHub: **Compare & pull request** | Deja el PR listo para la revisión conjunta |

**Dos auxilios.** Si editaste en main sin crear rama: `git checkout -b tu-rama` y tus cambios se van contigo. Si GitHub dice que tu rama está desactualizada: `git checkout main`, `git pull`, `git checkout tu-rama`, `git merge main`; si aparece un conflicto, no lo resuelvas solo, copia el mensaje en el chat.

## La revisión de cierre

Al final de cada bloque (cada 2 o 3 días, ver la guía día por día) los tres revisan juntos todos los PR en una llamada de 45 minutos. Un PR se fusiona con **las 2 aprobaciones de los otros dos y el CI en verde**. GitHub no lo obliga en repos privados gratuitos, así que lo cumplimos entre todos.

1. El dueño comparte pantalla: muestra qué cambió, `pytest` en verde y, si aplica, la tabla o figura que produjo.
2. Los otros miran **Files changed**: ¿tocó archivos ajenos?, ¿hay algo ajustado fuera del Pipeline?, ¿se leyó la prueba?
3. Comparan con el "Listo cuando" de esa persona en la guía día por día.
4. Si cumple, cada uno hace **Review changes → Approve**. Si algo falla, comentario en la línea exacta y no se fusiona hasta arreglarlo.
5. El dueño hace **Merge pull request**. Al final, Aleja crea la etiqueta del bloque (`git tag e2 && git push --tags`) y todos hacen el paso 1.

Se pide cambio solo por algo que no funciona, un archivo ajeno modificado o un riesgo de fuga. No se piden cambios de estilo ni de nombres de variables.

**Después de la fusión:** cada quien explica en 5 minutos una decisión de su PR a los otros dos. Es la forma más barata de llegar a la sustentación con "dominio de todo el equipo".

## Primer arranque en 15 minutos

Cada persona hace esto una vez, desde VS Code. Antes: VS Code con la extensión **Python** y **Jupyter** de Microsoft, **Git para Windows** (trae Git Bash) y **Python 3.12** con la casilla "Add python.exe to PATH".

- [ ] Aceptar la invitación al repositorio que llegó al correo de GitHub.
- [ ] En VS Code: **Terminal → New Terminal**, y en la flecha **⌵** junto al **+** elegir **Git Bash** (y **Select Default Profile → Git Bash** para que siempre abra así).
- [ ] Comprobar versiones: `python --version` (3.12) y `git --version`.
- [ ] Solo si nunca lo hiciste: `git config --global user.name "Tu Nombre"` y `git config --global user.email "el-correo-de-tu-github"`.
- [ ] Clonar: **Ctrl+Shift+P → Git: Clone**, pegar `https://github.com/jdrodriguez03/pronostico-amoblamiento-colombia`, elegir carpeta y **Open**.
- [ ] Crear el entorno: `python -m venv .venv` y aceptar el aviso de VS Code (o **Python: Select Interpreter → .venv**).
- [ ] Activarlo: `source .venv/Scripts/activate` (Mac: `source .venv/bin/activate`). Debe aparecer `(.venv)` al inicio de la línea.
- [ ] Instalar: `pip install -r requirements.txt` y luego `pip install -e .` (esto último hace que `from src...` funcione desde notebooks, scripts y pruebas).
- [ ] Probar: `pytest` debe salir en verde y `ruff check .` sin errores.
- [ ] En VS Code abrir cualquier notebook y elegir el kernel `.venv`.
- [ ] Abrir tu primer archivo (tabla "Quién cuida qué") y leer su primera línea.
- [ ] Escribir "listo" en el chat del equipo.

**Si algo falla:** copia el error completo de la terminal en el chat, no una descripción. No pases de 15 minutos con el mismo error de git.

## Preguntas abiertas para el arranque

- ¿Se confirman las decisiones del día 1 (predictores rezagados, logaritmo del objetivo, KNN)? Están en `config.py` y en la guía día por día.
- ¿El repositorio se hace público el 27 o se invita al profesor como colaborador?

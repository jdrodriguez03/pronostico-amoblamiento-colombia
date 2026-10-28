"""Resume la selección de Stepwise y Lasso en 5 folds por escenario.

Usa los hiperparámetros Lasso guardados en ``results/tablas/cv_resultados.csv`` y
ajusta ambos candidatos en los meses de entrenamiento de cada fold, con el mismo
``CVTemporalPorMes`` y los candidatos existentes. Al ejecutarse escribe la tabla de
estabilidad y añade su lectura a la bitácora. No calcula predicciones ni métricas.
"""

from __future__ import annotations

import pandas as pd

from src.config import ESCENARIOS, OBJETIVO, PREDICTORES_REZ1, RUTA_DOCS, RUTA_TABLAS
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.interpretacion import coeficientes
from src.evaluacion.particion import CVTemporalPorMes
from src.modelos.candidatos import CANDIDATOS, construir_candidato, prefijo
from src.modelos.stepwise import StepwiseSelector

MODELO_STEPWISE = "mlr_stepwise"
MODELO_LASSO = "lasso"
VARIABLES = [*PREDICTORES_REZ1, "covid"]
TOLERANCIA_CERO = 1e-10
N_AJUSTES = 15
ENCABEZADO_BITACORA = "## 2026-10-26 · Estabilidad de la selección"


def _alpha_seleccionado(cv: pd.DataFrame, escenario: str) -> float:
    """Recupera el alpha elegido por GridSearchCV, incluso si el CSV lo redondeó."""
    filas = cv.loc[(cv["escenario"] == escenario) & (cv["modelo"] == MODELO_LASSO)]
    if len(filas) != 5 or filas["fold"].nunique() != 5:
        raise ValueError(f"Se esperaban cinco folds Lasso para el escenario {escenario!r}")
    textos = filas["hiperparametros"].drop_duplicates().tolist()
    if len(textos) != 1:
        raise ValueError(f"El alpha Lasso no es único entre folds de {escenario!r}")

    pares = [parte.strip() for parte in textos[0].split(";")]
    alpha_texto = next((parte.removeprefix("alpha=") for parte in pares
                        if parte.startswith("alpha=")), None)
    if alpha_texto is None:
        raise ValueError(f"No se encontró alpha en {textos[0]!r}")

    # cv_resultados.csv conserva cuatro cifras significativas. Se empareja ese texto
    # con la rejilla original para recuperar el valor exacto que eligió GridSearchCV.
    rejilla = CANDIDATOS[MODELO_LASSO][2]["alpha"]
    coincidencias = [float(valor) for valor in rejilla if f"{valor:.4g}" == alpha_texto]
    if len(coincidencias) != 1:
        raise ValueError(f"El alpha {alpha_texto!r} no identifica unívocamente la rejilla")
    return coincidencias[0]


def _variable_fuente(nombre: str) -> str | None:
    """Quita el prefijo del ColumnTransformer para reconocer las seis variables."""
    for prefijo_columna in ("log__", "num__", "bin__"):
        if nombre.startswith(prefijo_columna):
            nombre = nombre.removeprefix(prefijo_columna)
            break
    return nombre if nombre in VARIABLES else None


def _seleccion_stepwise(ajustado) -> set[str]:
    """Devuelve predictores fuente elegidos por StepwiseSelector en un fold."""
    interno = ajustado.regressor_
    selector = interno.named_steps["sel"]
    if not isinstance(selector, StepwiseSelector):
        raise TypeError("El candidato mlr_stepwise no contiene StepwiseSelector")
    nombres = selector.feature_names_in_
    mascara = selector.get_support()
    return {
        variable
        for nombre, elegido in zip(nombres, mascara, strict=True)
        if elegido and (variable := _variable_fuente(str(nombre))) is not None
    }


def _no_nulos_lasso(ajustado) -> set[str]:
    """Devuelve predictores fuente con coeficiente Lasso distinto de cero."""
    tabla = coeficientes(ajustado)
    return {
        variable
        for nombre, valor in zip(tabla["variable"], tabla["coeficiente"], strict=True)
        if abs(float(valor)) > TOLERANCIA_CERO
        and (variable := _variable_fuente(str(nombre))) is not None
    }


def construir_tabla_estabilidad() -> pd.DataFrame:
    """Ajusta Stepwise y Lasso solo en los cinco entrenamientos de cada escenario."""
    ruta_cv = RUTA_TABLAS / "cv_resultados.csv"
    if not ruta_cv.is_file():
        raise FileNotFoundError(f"Falta la tabla con hiperparámetros: {ruta_cv}")
    cv_guardada = pd.read_csv(ruta_cv)

    conteos = {
        variable: {"stepwise": 0, "lasso": 0}
        for variable in VARIABLES
    }
    total = 0

    for escenario in ESCENARIOS:
        datos = cargar_entrenamiento(escenario)
        splitter = CVTemporalPorMes()
        alpha = _alpha_seleccionado(cv_guardada, escenario)

        for train_idx, _ in splitter.split(datos):
            entrenamiento = datos.iloc[train_idx]
            objetivo = entrenamiento[OBJETIVO]

            stepwise, _ = construir_candidato(MODELO_STEPWISE)
            stepwise.fit(entrenamiento, objetivo)
            for variable in _seleccion_stepwise(stepwise):
                conteos[variable]["stepwise"] += 1

            lasso, _ = construir_candidato(MODELO_LASSO)
            lasso.set_params(**{f"{prefijo()}alpha": alpha})
            lasso.fit(entrenamiento, objetivo)
            for variable in _no_nulos_lasso(lasso):
                conteos[variable]["lasso"] += 1

            total += 1

    if total != N_AJUSTES:
        raise ValueError(f"Se realizaron {total} pares fold-escenario; se esperaban {N_AJUSTES}")

    return pd.DataFrame([
        {
            "variable": variable,
            "stepwise_veces_seleccionada": conteos[variable]["stepwise"],
            "lasso_coeficiente_no_nulo": conteos[variable]["lasso"],
            "pares_fold_escenario": total,
        }
        for variable in VARIABLES
    ])


def _frase_interpretacion(tabla: pd.DataFrame) -> str:
    """Resume si la selección cambia entre folds o escenarios, sin fijar un umbral."""
    variables_que_varian = []
    for fila in tabla.itertuples(index=False):
        step = fila.stepwise_veces_seleccionada
        lasso = fila.lasso_coeficiente_no_nulo
        if 0 < step < N_AJUSTES or 0 < lasso < N_AJUSTES:
            variables_que_varian.append(
                f"{fila.variable} (Stepwise {step}/15; Lasso {lasso}/15)"
            )
    if variables_que_varian:
        return (
            "La selección cambia entre folds o escenarios para: "
            + ", ".join(variables_que_varian)
            + "."
        )
    return "La inclusión o exclusión de las seis variables es estable entre los 15 ajustes."


def _actualizar_bitacora(frase: str) -> None:
    """Añade o reemplaza la sección de estabilidad sin alterar las demás entradas."""
    ruta = RUTA_DOCS / "bitacora_resultados.md"
    texto = ruta.read_text(encoding="utf-8")
    bloque = f"{ENCABEZADO_BITACORA}\n\n{frase}\n"
    inicio = texto.find(ENCABEZADO_BITACORA)
    if inicio < 0:
        texto = texto.rstrip() + f"\n\n{bloque}"
    else:
        siguiente = texto.find("\n## ", inicio + len(ENCABEZADO_BITACORA))
        fin = len(texto) if siguiente < 0 else siguiente + 1
        texto = texto[:inicio] + bloque + texto[fin:]
    ruta.write_text(texto, encoding="utf-8")


def main() -> None:
    tabla = construir_tabla_estabilidad()
    RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_TABLAS / "estabilidad_seleccion.csv", index=False)
    frase = _frase_interpretacion(tabla)
    _actualizar_bitacora(frase)
    print(tabla.to_string(index=False))
    print(frase)


if __name__ == "__main__":
    main()

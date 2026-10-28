"""Analiza residuos de validacion y ajuste del modelo preliminar, sin abrir la prueba."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import INICIO_PRUEBA, OBJETIVO, RUTA_DOCS, RUTA_FIGURAS, RUTA_TABLAS
from src.datos.cargar import cargar_entrenamiento
from src.evaluacion.agregacion import agregar_nacional
from src.evaluacion.metricas import mad, mape
from src.evaluacion.particion import CVTemporalPorMes
from src.modelos.candidatos import construir_candidato, prefijo

MODELO = "ridge"
ESCENARIO = "central"
ENCABEZADO = "## 2026-10-27 · Análisis de errores del modelo preliminar"
RUTA_FIGURA = RUTA_FIGURAS / "analisis_errores_modelo.png"


def _parametros_guardados() -> dict:
    """Lee la configuración Ridge elegida en la tabla de validación central."""
    ruta = RUTA_TABLAS / "cv_resultados.csv"
    filas = pd.read_csv(ruta)
    filas = filas.loc[(filas["escenario"] == ESCENARIO) & (filas["modelo"] == MODELO)]
    if len(filas) != 5 or filas["fold"].nunique() != 5:
        raise ValueError("Se esperaban cinco filas de validación central para Ridge")
    textos = filas["hiperparametros"].drop_duplicates().tolist()
    if len(textos) != 1:
        raise ValueError("Los hiperparámetros Ridge no son únicos entre los folds centrales")

    parametros = {}
    for parte in textos[0].split(";"):
        nombre, valor = (pieza.strip() for pieza in parte.split("=", maxsplit=1))
        if nombre == "n_neighbors":
            parametros[nombre] = int(valor)
        elif nombre == "alpha":
            parametros[nombre] = float(valor)
    if set(parametros) != {"alpha"}:
        raise ValueError(f"No se pudieron leer los hiperparámetros Ridge: {textos[0]!r}")
    return parametros


def _pipeline(parametros: dict):
    pipeline, _ = construir_candidato(MODELO)
    pipeline.set_params(**{f"{prefijo()}{nombre}": valor for nombre, valor in parametros.items()})
    return pipeline


def _prediccion_nacional(df: pd.DataFrame, prediccion) -> pd.DataFrame:
    datos = df[["fecha", "departamento", "linea_codigo"]].copy()
    datos["observado"] = df[OBJETIVO].to_numpy(dtype=float)
    datos["predicho"] = prediccion
    observado = agregar_nacional(datos, "observado", linea_codigo=8)
    predicho = agregar_nacional(datos, "predicho", linea_codigo=8)
    resultado = pd.DataFrame({"observado": observado, "predicho": predicho})
    resultado.index = pd.to_datetime(resultado.index)
    resultado["residuo"] = resultado["observado"] - resultado["predicho"]
    return resultado.sort_index()


def _figura(validacion: pd.DataFrame, parametros: dict, metricas_covid: dict,
            mensual: pd.DataFrame) -> None:
    RUTA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    ax_residuos = fig.add_subplot(grid[0, :])
    ax_mensual = fig.add_subplot(grid[1, 0])
    ax_mape = fig.add_subplot(grid[1, 1])
    ax_mad = fig.add_subplot(grid[1, 2])

    ax_residuos.axhline(0, color="black", linewidth=1)
    ax_residuos.plot(validacion.index, validacion["residuo"], color="#3569a8",
                     marker="o", linewidth=1.2)
    mayores = validacion["residuo"].abs().nlargest(3).index
    for fecha in mayores:
        ax_residuos.annotate(fecha.strftime("%b %Y"),
                             (fecha, validacion.loc[fecha, "residuo"]),
                             xytext=(5, 7), textcoords="offset points", fontsize=8)
    ax_residuos.set(title="Residuos nacionales en validación (30 meses)",
                    ylabel="Observado − predicho", xlabel="Mes")

    etiquetas_mes = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                     "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    valores = mensual.reindex(range(1, 13))["error_absoluto_medio"]
    barras = ax_mensual.bar(range(1, 13), valores, color="#7a9e7e")
    ax_mensual.set(title="Error absoluto medio por mes calendario",
                   xlabel="Mes (n = 2 o 3)", ylabel="Índice")
    ax_mensual.set_xticks(range(1, 13), etiquetas_mes, rotation=45)
    ax_mensual.bar_label(barras, fmt="%.1f", fontsize=7, padding=2)

    etiquetas = ["Confinamiento", "Resto del entrenamiento"]
    for ax, metrica, titulo, unidad in [
        (ax_mape, "mape", "MAPE de ajuste", "%"),
        (ax_mad, "mad", "MAD de ajuste", "puntos"),
    ]:
        valores_metrica = [metricas_covid["covid"][metrica],
                           metricas_covid["no_covid"][metrica]]
        barras = ax.bar(etiquetas, valores_metrica, color=["#c87941", "#8aa6c1"])
        ax.set(title=titulo, ylabel=unidad)
        ax.tick_params(axis="x", labelrotation=20)
        ax.bar_label(barras, fmt="%.2f", padding=2)

    fig.suptitle(
        f"Modelo preliminar Ridge · alpha={parametros['alpha']:g} "
        "· entrenamiento hasta jul 2025"
    )
    fig.savefig(RUTA_FIGURA, dpi=180)
    plt.close(fig)


def _frase(validacion: pd.DataFrame, mensual: pd.DataFrame,
           metricas_covid: dict, mape_validacion_actual: float,
           mape_validacion_guardado: float) -> str:
    mayores = validacion.loc[validacion["residuo"].abs().nlargest(3).index]
    errores = "; ".join(
        f"{fecha:%b %Y}: {fila.residuo:+.2f} puntos"
        for fecha, fila in mayores.iterrows()
    )
    mes_mayor = int(mensual["error_absoluto_medio"].idxmax())
    nombres_meses = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ]
    nombre_mes = nombres_meses[mes_mayor - 1]
    covid = metricas_covid["covid"]
    no_covid = metricas_covid["no_covid"]
    if abs(mape_validacion_actual - mape_validacion_guardado) < 1e-6:
        estado_validacion = (
            f"El MAPE medio recalculado ({mape_validacion_actual:.6f} %) coincide con "
            f"el guardado ({mape_validacion_guardado:.6f} %)."
        )
    else:
        estado_validacion = (
            f"El MAPE medio recalculado ({mape_validacion_actual:.6f} %) no coincide con "
            f"el guardado ({mape_validacion_guardado:.6f} %); hay que reconciliar esa "
            "diferencia antes de consolidar las cifras del informe."
        )
    return (
        "El equipo eligió Ridge (alpha = 1000) como modelo preliminar: en central KNN "
        "queda 0,08 puntos porcentuales por debajo de Ridge, diferencia pequeña frente al "
        "error estándar pareado informado (0,88 puntos); se priorizaron la estabilidad de "
        "Ridge entre escenarios (9,02–9,14 %) y sus coeficientes sin cambios de signo. "
        "Los residuos de validación se recalcularon con el pipeline actual. "
        f"Los residuos son observado menos predicho: positivos indican subestimación y "
        f"negativos, sobreestimación. Los tres mayores errores absolutos fueron {errores}. "
        "El error absoluto medio por "
        f"mes calendario fue mayor en {nombre_mes} "
        f"({mensual.loc[mes_mayor, 'error_absoluto_medio']:.2f} "
        "puntos; solo 2 o 3 observaciones por mes), por lo que no basta para afirmar un patrón "
        "estacional recurrente. En ajuste dentro de muestra, los cinco meses con `covid=1` "
        f"tuvieron MAPE {covid['mape']:.2f} % y MAD {covid['mad']:.2f}, frente a "
        f"{no_covid['mape']:.2f} % y {no_covid['mad']:.2f} en los demás meses. Son residuos "
        "dentro de muestra, no errores de pronóstico. Los meses se conservaron y se identifican "
        "con `covid`; no se calculó distancia de Cook. "
        f"{estado_validacion}"
    )


def _actualizar_bitacora(frase: str) -> None:
    ruta = RUTA_DOCS / "bitacora_resultados.md"
    texto = ruta.read_text(encoding="utf-8")
    bloque = f"{ENCABEZADO}\n\n{frase}\n"
    inicio = texto.find(ENCABEZADO)
    if inicio < 0:
        texto = texto.rstrip() + f"\n\n{bloque}"
    else:
        siguiente = texto.find("\n## ", inicio + len(ENCABEZADO))
        fin = len(texto) if siguiente < 0 else siguiente + 1
        texto = texto[:inicio] + bloque + texto[fin:]
    ruta.write_text(texto, encoding="utf-8")


def main() -> None:
    datos = cargar_entrenamiento(ESCENARIO)
    fechas = pd.to_datetime(datos["fecha"])
    if fechas.max() >= pd.Timestamp(INICIO_PRUEBA):
        raise ValueError("El entrenamiento contiene fechas desde el inicio de la prueba")

    parametros = _parametros_guardados()
    predicciones = []
    with threadpool_limits(limits=1):
        for fold, (indices_entrena, indices_valida) in enumerate(
                CVTemporalPorMes().split(datos), start=1):
            entrenamiento = datos.iloc[indices_entrena]
            validacion = datos.iloc[indices_valida]
            modelo = _pipeline(parametros)
            modelo.fit(entrenamiento, entrenamiento[OBJETIVO])
            nacional = _prediccion_nacional(validacion, modelo.predict(validacion))
            nacional["fold"] = fold
            predicciones.append(nacional)

        modelo_ajuste = _pipeline(parametros)
        modelo_ajuste.fit(datos, datos[OBJETIVO])
        ajuste = _prediccion_nacional(datos, modelo_ajuste.predict(datos))

    validacion = pd.concat(predicciones).sort_index()
    if len(validacion) != 30 or not validacion.index.is_unique:
        raise ValueError("Se esperaban 30 meses únicos de validación")

    flags_covid = datos.groupby(pd.to_datetime(datos["fecha"]))["covid"].nunique()
    if (flags_covid != 1).any():
        raise ValueError("La bandera covid no es constante dentro de cada fecha")
    fechas_covid = set(pd.to_datetime(
        datos.loc[datos["covid"].eq(1), "fecha"].unique()
    ))
    mascara_covid = ajuste.index.isin(fechas_covid)
    metricas_covid = {}
    for nombre, mascara in [("covid", mascara_covid), ("no_covid", ~mascara_covid)]:
        parte = ajuste.loc[mascara]
        metricas_covid[nombre] = {
            "mape": mape(parte["observado"], parte["predicho"]),
            "mad": mad(parte["observado"], parte["predicho"]),
            "n": len(parte),
        }

    validacion["mes_calendario"] = validacion.index.month
    mensual = validacion.groupby("mes_calendario").agg(
        n=("residuo", "size"),
        error_absoluto_medio=("residuo", lambda serie: serie.abs().mean()),
    )
    mape_actual = mape(validacion["observado"], validacion["predicho"])
    guardados = pd.read_csv(RUTA_TABLAS / "cv_resultados.csv")
    guardados = guardados.loc[
        (guardados["escenario"] == ESCENARIO) & (guardados["modelo"] == MODELO)
    ].sort_values("fold")
    mape_guardado = float(guardados["mape"].mean())

    _figura(validacion, parametros, metricas_covid, mensual)
    frase = _frase(validacion, mensual, metricas_covid, mape_actual, mape_guardado)
    _actualizar_bitacora(frase)

    print(f"Figura: {RUTA_FIGURA}")
    print(f"Parámetros: {parametros}")
    print(f"MAPE validación actual/guardado: {mape_actual:.6f} / {mape_guardado:.6f}")
    print(f"COVID: {metricas_covid['covid']}; no COVID: {metricas_covid['no_covid']}")
    print(frase)


if __name__ == "__main__":
    main()

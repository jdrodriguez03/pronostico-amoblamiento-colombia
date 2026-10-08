# Dueño: Juanes
"""Estilo común de las figuras y guardado con nombre numerado.

Todos guardan las figuras del informe con guardar_figura(); así quedan en
resultados/figuras/ con el mismo estilo y tamaño.
"""

import matplotlib.pyplot as plt

from src.config import RUTA_FIGURAS


def aplicar_estilo() -> None:
    """Estilo base para todas las figuras (Juanes lo ajusta en el bloque 5)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def guardar_figura(fig, nombre: str) -> None:
    """Guarda fig en resultados/figuras/<nombre>.png, por ejemplo 'fig03_comparacion_cv'."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIGURAS / f"{nombre}.png", bbox_inches="tight")

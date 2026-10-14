"""Genera paneles y componentes de auditoría en data/processed."""

from src.config import ESCENARIOS, RUTA_PROCESSED
from src.datos.panel import _construir_panel_y_componentes


def main() -> None:
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for escenario in ESCENARIOS:
        panel, componentes = _construir_panel_y_componentes(escenario)
        panel.to_parquet(RUTA_PROCESSED / f"panel_{escenario}.parquet", index=False)
        componentes.to_parquet(
            RUTA_PROCESSED / f"componentes_{escenario}.parquet", index=False
        )


if __name__ == "__main__":
    main()

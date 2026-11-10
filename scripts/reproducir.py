# Dueño: Aleja
"""Corre todo en orden e imprime el MD5 de cada archivo de datos y tablas.

    python scripts/reproducir.py --hasta datos    # construye la base (simulación + predictores)
    python scripts/reproducir.py --hasta cv       # + scripts/entrenar.py --escenario todos
    python scripts/reproducir.py --hasta prueba   # + scripts/evaluar_prueba.py (con --forzar)

--forzar en la prueba: el script se niega a correr si prueba.csv ya existe; en un clon limpio
se usa para comprobar que se obtienen exactamente los mismos resultados (mismos hashes).
"""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.config import RUTA_INTERIM, RUTA_PROCESSED, RUTA_TABLAS  # noqa: E402

PASOS = {
    "datos": [["src/simulation/simular_base.py"], ["src/features/predictores.py"]],
    "cv": [["scripts/entrenar.py", "--escenario", "todos"]],
    "prueba": [["scripts/evaluar_prueba.py", "--escenario", "todos", "--forzar"]],
}
ORDEN = list(PASOS)


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def correr(comando: list[str]) -> None:
    print(f"\n>>> python {' '.join(comando)}", flush=True)
    inicio = time.time()
    subprocess.run([sys.executable, *comando], cwd=RAIZ, check=True)
    print(f"    listo en {time.time() - inicio:,.0f} s")


def imprimir_hashes() -> None:
    print("\nMD5 de los archivos generados:")
    for carpeta in (RUTA_INTERIM, RUTA_PROCESSED, RUTA_TABLAS):
        for ruta in sorted(carpeta.glob("*")):
            if ruta.is_file() and not ruta.name.startswith("."):
                print(f"{md5(ruta)}  {ruta.relative_to(RAIZ).as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hasta", choices=ORDEN, default="prueba")
    args = parser.parse_args(argv)
    inicio = time.time()
    for paso in ORDEN[: ORDEN.index(args.hasta) + 1]:
        for comando in PASOS[paso]:
            correr(comando)
    imprimir_hashes()
    print(f"\nTiempo total: {(time.time() - inicio) / 60:,.1f} min")


if __name__ == "__main__":
    main()

"""
Construccion de predictores y union con la base de la variable objetivo.

Uso (desde la raiz del repositorio):
    python src/features/predictores.py
        lee insumos de data/raw/ y base_*.csv de data/interim/;
        escribe base_final_*.csv en data/processed/ y predictores_*.csv en data/interim/
    python src/features/predictores.py <carpeta_insumos> <carpeta_salida>
        forma anterior: todo se lee y se escribe en <carpeta_salida>

Insumos esperados en data/raw/ (nombres pueden llevar prefijos):
    *anex-ELIC-SerieTipoBaseMun-*.xlsx   licencias de construccion por municipio (DANE)
    *anex-IMP-MensCapiArancel-*.xlsx     importaciones mensuales por capitulo (DANE)
    *anex-IPC-Indices-*.xlsx             IPC, series de empalme (DANE)
    *interes*.xlsx                       interes bancario corriente (Superfinanciera)
    *EOC*Hist*.pdf                       ICC historico (Fedesarrollo)
    base_bajo.csv, base_central.csv, base_alto.csv en data/interim/ (los genera simular_base.py)
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

# Rutas por defecto, relativas a la raiz del repositorio
RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"

INICIO, FIN = "2019-01-01", "2026-07-01"
MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
         "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
MES_ABR = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7,
           "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}
# Divipola -> dominio EMC
DOMINIO = {"05": "Antioquia", "08": "Atlantico", "11": "Bogota", "25": "Cundinamarca",
           "68": "Santander", "76": "Valle del Cauca"}


def buscar(carpeta, patron):
    archivos = sorted(Path(carpeta).glob(patron))
    if not archivos:
        raise FileNotFoundError(patron)
    return archivos[0]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- ICC (Fedesarrollo)
def icc(ruta):
    txt = subprocess.run(["pdftotext", "-layout", str(ruta), "-"], capture_output=True, text=True).stdout
    reg = []
    for linea in txt.splitlines():
        m = re.match(r"\s*([a-z]{3})-(\d{2})\s+(-?\d+,\d)\s+(-?\d+,\d)\s+(-?\d+,\d)", linea)
        if m:
            fecha = pd.Timestamp(2000 + int(m.group(2)), MES_ABR[m.group(1)], 1)
            reg.append((fecha, float(m.group(3).replace(",", "."))))
    s = pd.Series(dict(reg), name="icc").sort_index()
    return s[~s.index.duplicated()]


# ---------------------------------------------------------------- IPC (DANE)
def inflacion(ruta):
    rows = list(openpyxl.load_workbook(ruta, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    h = next(i for i, r in enumerate(rows) if r[0] == "Mes")
    anios = {j: int(str(x)[:4]) for j, x in enumerate(rows[h]) if x and str(x)[:4].isdigit()}
    reg = {}
    for r in rows[h + 1:h + 13]:
        mes = MESES.get(str(r[0]).strip().lower())
        if not mes:
            continue
        for j, a in anios.items():
            if np.isfinite(num(r[j])):
                reg[pd.Timestamp(a, mes, 1)] = num(r[j])
    ipc = pd.Series(reg).sort_index()
    return (ipc / ipc.shift(12) - 1).mul(100).rename("inflacion_anual")


# ---------------------------------------------------------------- Tasa (Superfinanciera)
def tasa(ruta):
    rows = list(openpyxl.load_workbook(ruta, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    reg = {}
    for r in rows[8:]:
        desde, hasta, val = r[2], r[3], num(r[4])
        if not np.isfinite(val):
            continue
        try:
            d, h = pd.Timestamp(desde), pd.Timestamp(hasta)
        except (ValueError, TypeError):
            continue
        if (h - d).days > 40:              # solo certificaciones mensuales
            continue
        reg[pd.Timestamp(d.year, d.month, 1)] = val * 100 if val < 1 else val
    return pd.Series(reg, name="tasa_consumo").sort_index()


# ---------------------------------------------------------------- Importaciones (DANE)
def importaciones(ruta):
    rows = list(openpyxl.load_workbook(ruta, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    h = next(i for i, r in enumerate(rows) if any(str(x).startswith("2026") for x in r if x))
    anios = {j: int(str(x)[:4]) for j, x in enumerate(rows[h]) if x and str(x)[:4].isdigit()}
    i0 = next(i for i, r in enumerate(rows) if r[0] and str(r[0]).strip().startswith("Muebles"))
    reg = {}
    for r in rows[i0:i0 + 12]:
        mes = MESES.get(str(r[1]).strip().lower())
        for j, a in anios.items():
            if mes and np.isfinite(num(r[j])):
                reg[pd.Timestamp(a, mes, 1)] = num(r[j])
    return pd.Series(reg, name="importaciones_muebles_usd_miles").sort_index()


# ---------------------------------------------------------------- Vivienda (DANE ELIC)
def vivienda(ruta):
    ws = openpyxl.load_workbook(ruta, read_only=True, data_only=True)["elic"]
    it = ws.iter_rows(values_only=True)
    hdr = [str(x) for x in next(it)]
    ix = {k: hdr.index(k) for k in ["año", "mes", "cod_depto", "destino", "area"]}
    reg = []
    for r in it:
        if r[ix["año"]] is None or int(r[ix["año"]]) < 2019 or int(r[ix["destino"]]) != 1:
            continue
        dep = DOMINIO.get(str(r[ix["cod_depto"]]).zfill(2), "Otros departamentos")
        reg.append((pd.Timestamp(int(r[ix["año"]]), int(r[ix["mes"]]), 1), dep, num(r[ix["area"]])))
    df = pd.DataFrame(reg, columns=["fecha", "departamento", "area"])
    return df.groupby(["fecha", "departamento"]).area.sum().rename("area_vivienda_m2").reset_index()


def main(insumos=RAW, interim=INTERIM, procesada=PROCESSED):
    interim, procesada = Path(interim), Path(procesada)
    procesada.mkdir(parents=True, exist_ok=True)
    fechas = pd.date_range(INICIO, FIN, freq="MS")
    nac = pd.concat([icc(buscar(insumos, "*EOC*.pdf")),
                     inflacion(buscar(insumos, "*anex-IPC-Indices*.xlsx")),
                     tasa(buscar(insumos, "*interes*.xlsx")),
                     importaciones(buscar(insumos, "*anex-IMP-MensCapiArancel*.xlsx"))], axis=1)
    # rezago de un mes (el dato del mes t no se conoce al pronosticar t)
    for c in list(nac.columns):
        nac[f"{c}_rez1"] = nac[c].shift(1)
    nac = nac.reindex(fechas)
    nac.index.name = "fecha"
    # Aislamiento preventivo obligatorio nacional: 25-mar-2020 a 31-ago-2020
    nac["covid"] = ((nac.index >= "2020-04-01") & (nac.index <= "2020-08-01")).astype(int)
    nac = nac.reset_index()

    viv = vivienda(buscar(insumos, "*anex-ELIC-SerieTipoBaseMun*.xlsx"))
    viv = viv.sort_values(["departamento", "fecha"])
    viv["area_vivienda_m2_rez1"] = viv.groupby("departamento").area_vivienda_m2.shift(1)
    # El rezago de enero 2019 queda vacio: antes de 2019 la cobertura de ELIC era menor y no es comparable
    viv = viv[viv.fecha.between(INICIO, FIN)]

    resumen = []
    for esc in ["bajo", "central", "alto"]:
        base = pd.read_csv(interim / f"base_{esc}.csv", parse_dates=["fecha"])
        final = base.merge(nac, on="fecha", how="left").merge(viv, on=["fecha", "departamento"], how="left")
        final.to_csv(procesada / f"base_final_{esc}.csv", index=False, float_format="%.4f")
        resumen.append((esc, final.shape, final.isna().sum()[final.isna().sum() > 0].to_dict()))
    nac.to_csv(interim / "predictores_nacionales.csv", index=False, float_format="%.4f")
    viv.to_csv(interim / "predictores_vivienda_departamental.csv", index=False, float_format="%.2f")
    for r in resumen:
        print(r)
    print(nac.describe().T.round(2).to_string())
    print(viv.groupby("departamento").area_vivienda_m2.describe().round(0).to_string())


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], sys.argv[2])
    else:
        main()

# Fuentes de los datos crudos

Dueño: Julián. Archivos tal como se descargaron; nadie los edita. Pendiente: URL exacta y fecha de descarga.

| Archivo | Fuente | Contenido | Lo lee | URL | Descargado |
| --- | --- | --- | --- | --- | --- |
| `anex-EMC-ComercioMinorista-jul2026.xlsx` | DANE, Encuesta Mensual de Comercio, anexo nacional | Índices reales por línea (hoja 2.2) y actividades 474/475 (hoja 2.4) | `simular_base.py` | | |
| `anex-EMC-ComercioalpormenorDep-jul2026.xlsx` | DANE, EMC, anexo departamental | Contribuciones (1.1), participación del grupo 4741–4759 (2.1) e índices por dominio (3.2) | `simular_base.py` | | |
| `anex-ELIC-SerieTipoBaseMun-jul2026.xlsx` | DANE, Estadísticas de Licencias de Construcción | Área aprobada por municipio y destino (hoja `elic`) | `predictores.py` | | |
| `anex-ELIC-SerieHist1000Mun-jul2026.xlsx` | DANE, ELIC, serie histórica 1.000 municipios | Referencia; no lo lee ningún script | — | | |
| `anex-IMP-MensCapiArancel-jul2026.xlsx` | DANE, importaciones por capítulo del arancel | Capítulo 94 (muebles), miles de USD CIF | `predictores.py` | | |
| `anex-IPC-Indices-ago2026.xlsx` | DANE, IPC, series de empalme | Índice total mensual | `predictores.py` | | |
| `interes.xlsx` | Superintendencia Financiera | Interés bancario corriente, crédito de consumo y ordinario | `predictores.py` | | |
| `EOC_Julio_2026_Hit_rico.pdf` | Fedesarrollo, Encuesta de Opinión del Consumidor | ICC histórico nacional (se lee con `pdftotext`) | `predictores.py` | | |

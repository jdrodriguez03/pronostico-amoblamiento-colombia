# Verificacion de la base construida

Semilla: 42 | Meses: 91 (2019-01 a 2026-07) | Departamentos: 7 | Lineas: 8

Pesos departamentales (base 2019): Antioquia 17.0%, Atlantico 5.7%, Bogota 22.6%, Cundinamarca 5.0%, Santander 4.3%, Valle del Cauca 12.1%, Otros departamentos 33.3%
Verificacion contable de pesos vs actividades 474/475: error maximo 8.7e-14 %

## Escenario bajo (desviacion de eta = 2%)
1. Integridad: 5096 registros | vacios 0 | no positivos 0
2. Conservacion nacional: error maximo 4.4e-14 %
3. Ruido recuperado: desviacion 1.9 % (objetivo 2 %) | autocorrelacion 0.56 (objetivo 0.6)
4. Fidelidad regional (muebles vs grupo real del depto): correlacion 0.85 a 0.93
5. Patrones: muebles nacional abr-2020 = 41.3 (real 41.3)

## Escenario central (desviacion de eta = 7%)
1. Integridad: 5096 registros | vacios 0 | no positivos 0
2. Conservacion nacional: error maximo 4.4e-14 %
3. Ruido recuperado: desviacion 6.8 % (objetivo 7 %) | autocorrelacion 0.56 (objetivo 0.6)
4. Fidelidad regional (muebles vs grupo real del depto): correlacion 0.82 a 0.90
5. Patrones: muebles nacional abr-2020 = 41.3 (real 41.3)

## Escenario alto (desviacion de eta = 13%)
1. Integridad: 5096 registros | vacios 0 | no positivos 0
2. Conservacion nacional: error maximo 4.4e-14 %
3. Ruido recuperado: desviacion 12.6 % (objetivo 13 %) | autocorrelacion 0.58 (objetivo 0.6)
4. Fidelidad regional (muebles vs grupo real del depto): correlacion 0.70 a 0.84
5. Patrones: muebles nacional abr-2020 = 41.3 (real 41.3)

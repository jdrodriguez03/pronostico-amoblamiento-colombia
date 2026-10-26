# Dueño: Julián
"""StepwiseSelector: selección stepwise bidireccional (alfa = 0,25, como el artículo).

Reproduce el procedimiento de Minitab que usó el artículo base, pero como un
transformador de scikit-learn. Así va DENTRO del Pipeline (regla de oro 2): en cada
fold de la validación cruzada elige variables mirando solo los meses de entrenamiento.

Cómo funciona (una vuelta del ciclo):

1. Paso hacia adelante: prueba agregar, una por una, cada variable que no está en el
   modelo. Se queda con la de menor p-valor, si ese p-valor es < alfa_entrada.
2. Paso hacia atrás: con el modelo actual, mira las variables no forzadas. Si la de
   mayor p-valor supera alfa_salida, la saca (una variable que entró antes puede dejar
   de ser útil cuando entra otra correlacionada con ella).
3. Repite hasta que en una vuelta no entre ni salga nada.

Las variables "forzadas" (las dummies de mes, departamento y línea) están siempre en el
modelo: son la estacionalidad y los efectos fijos, no se someten a selección.

Limitación que se declara en docs/decisiones.md: las 56 filas de un mes comparten los
predictores nacionales, así que no son independientes y los p-valores de OLS salen más
pequeños de lo que deberían (optimistas). Por eso el stepwise se compara con Lasso.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from src.config import ALFA_STEPWISE

# Tope de vueltas del ciclo. Con alfa_entrada <= alfa_salida el ciclo siempre termina,
# pero el tope evita un bucle infinito si alguien elige alfas raros.
MAX_VUELTAS = 100


class StepwiseSelector(SelectorMixin, BaseEstimator):
    """Selector stepwise bidireccional con p-valores de OLS (statsmodels).

    Parameters
    ----------
    alfa_entrada : float
        Una variable entra si su p-valor es menor que este valor (0,25 en el artículo).
    alfa_salida : float
        Una variable sale si su p-valor es mayor que este valor (0,25 en el artículo).
    forzadas : None, lista de int o lista de str
        Columnas que siempre quedan en el modelo.
        - None: ninguna forzada.
        - Enteros: posiciones de las columnas (0, 1, 2...).
        - Textos: prefijos de nombre. Requiere que X llegue como DataFrame, por ejemplo
          ("cat__",) fuerza todas las dummies que crea el ColumnTransformer de Juanes.

    Attributes (después de fit)
    ----------
    seleccion_ : np.ndarray de bool, True en las columnas elegidas (forzadas incluidas).
    historial_ : list[tuple[str, str, float]], cada entrada o salida: (accion, columna, p).
    """

    def __init__(self, alfa_entrada=ALFA_STEPWISE, alfa_salida=ALFA_STEPWISE, forzadas=None):
        # scikit-learn exige que __init__ solo guarde los parámetros tal cual: así
        # funcionan clone(), get_params() y GridSearchCV.
        self.alfa_entrada = alfa_entrada
        self.alfa_salida = alfa_salida
        self.forzadas = forzadas

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------
    def fit(self, X, y):
        if self.alfa_entrada > self.alfa_salida:
            raise ValueError("alfa_entrada debe ser <= alfa_salida, o el ciclo no termina")

        # validate_data revisa X y y, y guarda n_features_in_ y, si X es DataFrame,
        # feature_names_in_ (los nombres que usaremos para las forzadas y el historial).
        X, y = validate_data(self, X, y, dtype=float, y_numeric=True)
        n = X.shape[1]
        nombres = self._nombres(n)
        forzadas = self._indices_forzadas(nombres)

        en_modelo = list(forzadas)
        candidatas = [j for j in range(n) if j not in forzadas]
        self.historial_ = []

        for _ in range(MAX_VUELTAS):
            cambio = False

            # 1. Hacia adelante: la mejor candidata que aún no está.
            fuera = [j for j in candidatas if j not in en_modelo]
            if fuera:
                p_nuevas = {j: self._pvalores(X, y, en_modelo + [j])[-1] for j in fuera}
                mejor = min(p_nuevas, key=p_nuevas.get)
                if p_nuevas[mejor] < self.alfa_entrada:
                    en_modelo.append(mejor)
                    self.historial_.append(("entra", nombres[mejor], p_nuevas[mejor]))
                    cambio = True

            # 2. Hacia atrás: la peor de las no forzadas que están.
            libres = [j for j in en_modelo if j not in forzadas]
            if libres:
                p_modelo = self._pvalores(X, y, en_modelo)
                p_libres = {j: p_modelo[en_modelo.index(j)] for j in libres}
                peor = max(p_libres, key=p_libres.get)
                if p_libres[peor] > self.alfa_salida:
                    en_modelo.remove(peor)
                    self.historial_.append(("sale", nombres[peor], p_libres[peor]))
                    cambio = True

            if not cambio:
                break

        self.seleccion_ = np.zeros(n, dtype=bool)
        self.seleccion_[en_modelo] = True
        return self

    def _get_support_mask(self):
        # SelectorMixin usa esta máscara para transform() y get_feature_names_out().
        check_is_fitted(self, "seleccion_")
        return self.seleccion_

    # ------------------------------------------------------------------
    # Ayudantes
    # ------------------------------------------------------------------
    @staticmethod
    def _pvalores(X, y, columnas):
        """p-valores de OLS con intercepto para las columnas dadas, en ese orden."""
        if not columnas:
            return np.array([])
        Xs = sm.add_constant(X[:, columnas], has_constant="add")
        resultado = sm.OLS(y, Xs).fit()
        return np.asarray(resultado.pvalues)[1:]  # el 0 es la constante

    def _nombres(self, n):
        if hasattr(self, "feature_names_in_"):
            return [str(c) for c in self.feature_names_in_]
        return [f"x{j}" for j in range(n)]

    def _indices_forzadas(self, nombres):
        if self.forzadas is None:
            return []
        forzadas = list(self.forzadas)
        if all(isinstance(f, str) for f in forzadas):
            if not hasattr(self, "feature_names_in_"):
                raise ValueError(
                    "forzadas por nombre requiere que X sea un DataFrame "
                    "(pipeline.set_output(transform='pandas'))"
                )
            return [j for j, c in enumerate(nombres) if c.startswith(tuple(forzadas))]
        return sorted({int(f) for f in forzadas})


def tabla_historial(selector: StepwiseSelector) -> pd.DataFrame:
    """Entradas y salidas del stepwise en orden, para la bitácora y el informe."""
    check_is_fitted(selector, "seleccion_")
    return pd.DataFrame(selector.historial_, columns=["accion", "variable", "p_valor"])

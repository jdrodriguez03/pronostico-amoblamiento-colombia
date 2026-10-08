# Dueño: Juanes
"""ColumnTransformer: codificar categóricas y escalar numéricas. Día 4 de la guía.

Va SIEMPRE dentro del Pipeline (regla de oro 2): se ajusta en cada fold.
"""

from sklearn.compose import ColumnTransformer


def crear_preprocesador() -> ColumnTransformer:
    """OneHotEncoder para mes, departamento y linea; StandardScaler para los numéricos."""
    raise NotImplementedError("Juanes, día 4")

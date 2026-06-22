"""
core/calif_config.py
Configuración editable de las calificaciones:
  - mapeo número -> letra (10->AAA, 9->AA, ...)
  - cortes de las 3 clases (qué números caen en Buena / Media / Riesgo)

Se usa tanto para mostrar la letra como para agrupar en 3 clases al entrenar.
"""

# Mapeo por RANGOS -> letra. Cada letra cubre un intervalo [min, max) de calificación
# (min cerrado, max abierto; min null = -inf, max null = +inf). Cubre todo el 0-10,
# así el 3, 7 o cualquier valor cae en su letra sin listarlos uno por uno.
LETRAS_DEFAULT = [
    {"letra": "AAA", "min": 10, "max": None},   # 10
    {"letra": "AA",  "min": 9,  "max": 10},     # 9
    {"letra": "A",   "min": 8,  "max": 9},      # 8   (7 NO llega -> baja)
    {"letra": "BBB", "min": 6,  "max": 8},      # 6, 7   (7 redondea abajo a BBB)
    {"letra": "BB",  "min": 5,  "max": 6},      # 5
    {"letra": "B",   "min": 4,  "max": 5},      # 4   (3 NO llega -> baja)
    {"letra": "CCC", "min": 2,  "max": 4},      # 2, 3   (3 redondea abajo a CCC)
    {"letra": "CC",  "min": 1,  "max": 2},      # 1
    {"letra": "C",   "min": None, "max": 1},    # 0
]

# Cortes de las 3 clases por defecto: Buena si >= 9, Media si >= 6, Riesgo el resto.
# Se define con umbrales mínimos (inclusive). El orden importa: de mayor a menor.
CORTES_DEFAULT = {
    "Buena": 9,    # valor >= 9  -> Buena
    "Media": 6,    # valor >= 6  -> Media
    "Riesgo": 0,   # valor >= 0  -> Riesgo (resto)
}

# orden de las clases de mejor a peor
ORDEN_CLASES = ["Buena", "Media", "Riesgo"]


def numero_a_letra(valor, letras=None):
    """Devuelve la letra cuyo rango [min,max) contiene al valor."""
    letras = letras or LETRAS_DEFAULT
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    # compatibilidad: si llega el formato viejo (dict número->letra), usarlo
    if isinstance(letras, dict):
        return letras.get(str(int(v)), str(int(v)))
    for r in letras:
        lo = r.get("min"); hi = r.get("max")
        if (lo is None or v >= lo) and (hi is None or v < hi):
            return r.get("letra", str(v))
    return str(int(v))


def agrupar_con_cortes(valor, cortes=None):
    """Agrupa un valor numérico en Buena/Media/Riesgo según los cortes (umbral mínimo)."""
    cortes = cortes or CORTES_DEFAULT
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "Riesgo"
    # evaluar de mejor a peor: la primera clase cuyo umbral se cumpla
    for clase in ORDEN_CLASES:
        if clase in cortes and v >= cortes[clase]:
            return clase
    return "Riesgo"

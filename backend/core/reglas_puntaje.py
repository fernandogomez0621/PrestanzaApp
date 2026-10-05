"""
core/reglas_puntaje.py
Motor de reglas por ESCALONES (rango -> valor fijo) totalmente configurable para
el puntaje del deudor principal. Reemplaza las funciones por trozos hardcodeadas
por una estructura de datos editable y guardable.

Estructura de la config (JSON):
{
  "categorias": {            # factores categóricos (texto -> puntaje)
     "educacion": {"sin estudio":0,"primaria":0,...,"_default":0}
  },
  "escalones": {             # factores numéricos por rangos
     "credit_score": [
        {"min": null, "max": 450, "valor": 0},     # min null = -infinito
        {"min": 450,  "max": 550, "valor": 1},
        {"min": 550,  "max": 800, "valor": 6},
        {"min": 800,  "max": 900, "valor": 11},
        {"min": 900,  "max": null,"valor": 12}      # max null = +infinito
     ], ...
  }
}
Un escalón aplica si  min <= x < max  (min null = sin límite inferior; max null = sin límite superior).
"""

# Configuración por defecto: réplica en escalones de las funciones originales.
# (Las que eran lineales se aproximan al punto medio del tramo, como pediste.)
CONFIG_DEFAULT = {
    "categorias": {
        "educacion": {
            "sin estudio": 0, "primaria": 0, "bachillerato": 0,
            "técnico profesional": 5, "tecnológico": 5,
            "pregrado": 8,
            "especialización": 10, "maestría": 10, "doctorado": 10,
            "_default": 0,
        }
    },
    "escalones": {
        "credit_score": [
            {"min": None, "max": 450, "valor": 0},
            {"min": 450, "max": 550, "valor": 1},
            {"min": 550, "max": 800, "valor": 6},      # antes lineal 1→11; escalón = 6 (medio)
            {"min": 800, "max": 900, "valor": 11},
            {"min": 900, "max": None, "valor": 12},
        ],
        "equity": [
            {"min": None, "max": 2, "valor": 0},
            {"min": 2, "max": 10, "valor": 5},          # antes lineal 0→10; escalón = 5
            {"min": 10, "max": None, "valor": 10},
        ],
        "quote": [
            {"min": None, "max": 5, "valor": 0},
            {"min": 5, "max": 10, "valor": 6},
            {"min": 10, "max": 20, "valor": 8},
            {"min": 20, "max": None, "valor": 10},
        ],
        "creditos_cerrados": [
            {"min": None, "max": 0.0001, "valor": 0},
            {"min": 0.0001, "max": 10, "valor": 5},
            {"min": 10, "max": None, "valor": 10},
        ],
        "similar_loans": [
            {"min": None, "max": 1, "valor": 0},
            {"min": 1, "max": 3, "valor": 4},
            {"min": 3, "max": 5, "valor": 8},
            {"min": 5, "max": None, "valor": 10},
        ],
        "bank": [
            {"min": None, "max": 0.0001, "valor": 0},
            {"min": 0.0001, "max": 5, "valor": 6},
            {"min": 5, "max": 10, "valor": 10},
            {"min": 10, "max": None, "valor": 12},
        ],
        "propiedades": [
            {"min": None, "max": 1, "valor": 0},
            {"min": 1, "max": 3, "valor": 4},
            {"min": 3, "max": 5, "valor": 8},
            {"min": 5, "max": None, "valor": 10},
        ],
        # negativos: el valor se RESTA (se aplica el signo en el cálculo del total)
        "demandas": [
            {"min": None, "max": 0.0001, "valor": 0},
            {"min": 0.0001, "max": 3, "valor": 1},
            {"min": 3, "max": 10, "valor": 5},
            {"min": 10, "max": None, "valor": 10},
        ],
        "demandas_dinero": [
            {"min": None, "max": 0.0001, "valor": 0},
            {"min": 0.0001, "max": 2, "valor": 3},
            {"min": 2, "max": 6, "valor": 8},
            {"min": 6, "max": None, "valor": 15},
        ],
    },
}

# factores cuyo puntaje se resta en el total
FACTORES_NEGATIVOS = {"demandas", "demandas_dinero"}

# etiquetas legibles
LABELS_FACTORES = {
    "educacion": "Nivel de educación", "credit_score": "Score crediticio",
    "equity": "Patrimonio / equity", "quote": "Capacidad de pago (cuota)",
    "creditos_cerrados": "Créditos cerrados", "similar_loans": "Historial crédito similar",
    "bank": "Factores bancarios", "propiedades": "Propiedades",
    "demandas": "Demandas (general) — resta", "demandas_dinero": "Demandas por dinero — resta",
}


def evaluar_escalon(valor, escalones):
    """Devuelve el valor del escalón que contiene a `valor` (min <= x < max)."""
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return 0
    for esc in escalones:
        lo = esc.get("min"); hi = esc.get("max")
        if (lo is None or x >= lo) and (hi is None or x < hi):
            return float(esc.get("valor", 0))
    return 0


def evaluar_categoria(valor, mapa):
    return float(mapa.get(str(valor), mapa.get("_default", 0)))

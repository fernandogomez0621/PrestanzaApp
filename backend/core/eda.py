"""
core/eda.py
Análisis exploratorio (EDA) robusto de las variables del modelo frente a la
calificación del crédito. Usa el deudor principal según el modo elegido
(titular o mejor perfil), exactamente igual que el entrenamiento.

Entrega, por horizonte:
  - resumen por variable: media/mediana/std por clase (Buena/Media/Riesgo),
    correlación (Spearman) con la calificación, y poder de separación
    (eta² entre clases) para rankear qué variables discriminan mejor.
  - distribución de cada variable por clase (cuartiles para boxplots).
  - promedio de cada variable por cada calificación numérica.
  - matriz de correlación entre variables (para ver redundancias).
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import procesar_horizonte, features_modelo
from clasificador_riesgo import agrupar_calificacion
from scipy import stats

ORDEN = ['Buena', 'Media', 'Riesgo']


def _eta_cuadrado(grupos):
    """Poder de separación de una variable entre clases (0-1). eta² alto = separa bien."""
    valores = [g for g in grupos if len(g) > 0]
    if len(valores) < 2:
        return 0.0
    todos = np.concatenate(valores)
    media_global = todos.mean()
    ss_total = ((todos - media_global) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_entre = sum(len(g) * (g.mean() - media_global) ** 2 for g in valores)
    return float(np.clip(ss_entre / ss_total, 0, 1))


def analizar(datapoints_path, calificaciones_path, modo_seleccion='titular',
             pesos=None, reglas=None, cortes=None):
    datos = procesar_horizonte(datapoints_path, calificaciones_path,
                               modo_seleccion=modo_seleccion, pesos=pesos, reglas=reglas)
    X, y = features_modelo(datos)
    clase = y.apply(lambda v: agrupar_calificacion(v, cortes))
    cal = y.astype(float)

    variables = list(X.columns)
    n = len(X)

    # --- resumen por variable ---
    resumen = []
    for v in variables:
        col = pd.to_numeric(X[v], errors='coerce')
        grupos = {c: col[clase == c].dropna().values for c in ORDEN}
        # correlación de Spearman con la calificación (robusta a no-linealidad/outliers)
        try:
            rho, pval = stats.spearmanr(col, cal, nan_policy='omit')
        except Exception:
            rho, pval = np.nan, np.nan
        eta2 = _eta_cuadrado([grupos[c] for c in ORDEN])
        fila = {
            'variable': v,
            'spearman': None if pd.isna(rho) else round(float(rho), 3),
            'p_valor': None if pd.isna(pval) else round(float(pval), 4),
            'significativa': bool(pval < 0.05) if not pd.isna(pval) else False,
            'eta2': round(eta2, 3),
            'media_global': round(float(col.mean()), 3),
        }
        for c in ORDEN:
            g = grupos[c]
            fila[f'media_{c}'] = round(float(np.mean(g)), 3) if len(g) else None
        resumen.append(fila)
    # ordenar por poder de separación
    resumen.sort(key=lambda d: -d['eta2'])

    # --- distribución por clase (cuartiles para boxplot) de las top variables ---
    top_vars = [r['variable'] for r in resumen[:12]]
    boxplots = {}
    for v in top_vars:
        col = pd.to_numeric(X[v], errors='coerce')
        boxplots[v] = {}
        for c in ORDEN:
            g = col[clase == c].dropna().values
            if len(g):
                boxplots[v][c] = {
                    'min': round(float(np.min(g)), 3),
                    'q1': round(float(np.percentile(g, 25)), 3),
                    'mediana': round(float(np.percentile(g, 50)), 3),
                    'q3': round(float(np.percentile(g, 75)), 3),
                    'max': round(float(np.max(g)), 3),
                    'n': int(len(g)),
                }
    # --- promedio por calificación numérica ---
    prom_por_calif = {}
    for v in top_vars:
        col = pd.to_numeric(X[v], errors='coerce')
        serie = col.groupby(cal).mean()
        prom_por_calif[v] = {str(int(k)): round(float(val), 3) for k, val in serie.items() if not pd.isna(val)}

    # --- matriz de correlación entre variables (Spearman) ---
    Xnum = X.apply(pd.to_numeric, errors='coerce')
    corr = Xnum.corr(method='spearman').round(2)
    matriz = {'variables': variables, 'valores': corr.fillna(0).values.tolist()}

    # --- HISTOGRAMAS de variables cuantitativas (global y por clase) ---
    histogramas = {}
    for v in variables:
        col = pd.to_numeric(X[v], errors='coerce').dropna()
        if len(col) == 0 or col.nunique() <= 1:
            continue
        # bordes de bins comunes (10 bins, robustos a outliers vía percentiles)
        lo, hi = np.percentile(col, 1), np.percentile(col, 99)
        if lo == hi:
            lo, hi = col.min(), col.max()
        if lo == hi:
            continue
        bins = np.linspace(lo, hi, 11)
        centros = [round(float((bins[i] + bins[i + 1]) / 2), 2) for i in range(len(bins) - 1)]
        total, _ = np.histogram(col.clip(lo, hi), bins=bins)
        por_clase = {}
        for c in ORDEN:
            cc = pd.to_numeric(X[v][clase == c], errors='coerce').dropna()
            h, _ = np.histogram(cc.clip(lo, hi), bins=bins)
            por_clase[c] = [int(x) for x in h]
        histogramas[v] = {
            'centros': centros,
            'total': [int(x) for x in total],
            'por_clase': por_clase,
        }

    # --- DISTRIBUCIÓN de la calificación (histograma del target) ---
    hist_calif = cal.round().astype(int).value_counts().sort_index()
    distribucion_calificacion = {str(int(k)): int(v) for k, v in hist_calif.items()}

    # --- DISTRIBUCIONES CATEGÓRICAS (educación, etc.) global y por clase ---
    categoricas = {}
    clase_idx = clase.reset_index(drop=True)
    for col_cat in ['debtor_level_of_education']:
        if col_cat in datos.columns:
            serie = datos[col_cat].reset_index(drop=True).astype(str)
            global_ = serie.value_counts().to_dict()
            por_clase = {}
            for c in ORDEN:
                sub = serie[clase_idx == c]
                por_clase[c] = sub.value_counts().to_dict()
            categoricas[col_cat] = {
                'global': {str(k): int(v) for k, v in global_.items()},
                'por_clase': {c: {str(k): int(v) for k, v in d.items()} for c, d in por_clase.items()},
            }

    # --- distribución de clases ---
    dist_clases = clase.value_counts().to_dict()

    return {
        'n': int(n),
        'modo_seleccion': modo_seleccion,
        'variables': variables,
        'resumen': resumen,
        'boxplots': boxplots,
        'promedio_por_calificacion': prom_por_calif,
        'matriz_correlacion': matriz,
        'histogramas': histogramas,
        'distribucion_calificacion': distribucion_calificacion,
        'categoricas': categoricas,
        'distribucion_clases': dist_clases,
        'top_variables': top_vars,
    }

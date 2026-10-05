"""
core/multimodelo.py
Compara varios algoritmos para una tarea de clasificación, elige el mejor por
F1-macro (validación cruzada estratificada) y reporta para CADA candidato:
  - métricas globales (f1_macro, accuracy, balanced_accuracy)
  - métricas por clase (precision, recall, f1)
  - matriz de confusión (con validación cruzada, sin fuga de datos)
  - importancia de variables (coeficientes o feature_importances_ o permutación)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.inspection import permutation_importance


def _catalogo(n_clases):
    """Algoritmos a comparar. Hiperparámetros conservadores para n pequeño."""
    return {
        'Regresión Logística': lambda: LogisticRegression(class_weight='balanced', max_iter=3000),
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight='balanced', random_state=42),
        'Gradient Boosting': lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=min(7, max(3, n_clases + 2))),
    }


def _importancia(nombre, pipe, X, y, feature_names):
    """Importancia de variables del algoritmo entrenado (con dirección si aplica)."""
    clf = pipe.named_steps['clf']
    try:
        if hasattr(clf, 'coef_'):
            coefs = clf.coef_
            if coefs.ndim > 1:
                imp = np.abs(coefs).mean(axis=0)
                signo = np.sign(coefs.mean(axis=0))
            else:
                imp = np.abs(coefs); signo = np.sign(coefs)
            filas = [{'variable': f, 'importancia': round(float(imp[i]), 4),
                      'direccion': ('+' if signo[i] >= 0 else '−')} for i, f in enumerate(feature_names)]
        elif hasattr(clf, 'feature_importances_'):
            fi = clf.feature_importances_
            filas = [{'variable': f, 'importancia': round(float(fi[i]), 4), 'direccion': None}
                     for i, f in enumerate(feature_names)]
        else:
            # KNN u otros sin importancia nativa -> permutación
            r = permutation_importance(pipe, X, y, n_repeats=10, random_state=42, scoring='f1_macro')
            filas = [{'variable': f, 'importancia': round(float(r.importances_mean[i]), 4), 'direccion': None}
                     for i, f in enumerate(feature_names)]
    except Exception:
        filas = []
    filas.sort(key=lambda d: -d['importancia'])
    return filas[:15]


def comparar(X, y, feature_names, orden_clases=None, scoring='f1_macro'):
    """Compara algoritmos y devuelve el mejor + el reporte de todos."""
    Xv = X.values if hasattr(X, 'values') else np.asarray(X)
    y = np.asarray(y)
    clases = orden_clases if orden_clases is not None else sorted(set(y), key=str)
    n = len(y)
    n_splits = max(2, min(5, np.min(np.bincount(
        [list(clases).index(v) if v in clases else 0 for v in y])) if len(clases) else 3))
    n_splits = min(5, max(2, n_splits))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    catalogo = _catalogo(len(clases))
    reportes = []
    mejor = None

    for nombre, fabrica in catalogo.items():
        pipe = Pipeline([('sc', StandardScaler()), ('clf', fabrica())])
        try:
            # predicciones por CV (para matriz de confusión honesta)
            y_pred = cross_val_predict(pipe, Xv, y, cv=cv)
            f1m = float(f1_score(y, y_pred, average='macro'))
            acc = float(accuracy_score(y, y_pred))
            bacc = float(balanced_accuracy_score(y, y_pred))
            cm = confusion_matrix(y, y_pred, labels=clases).tolist()
            prec, rec, f1c, sup = precision_recall_fscore_support(y, y_pred, labels=clases, zero_division=0)
            por_clase = [{'clase': str(clases[i]), 'precision': round(float(prec[i]), 3),
                          'recall': round(float(rec[i]), 3), 'f1': round(float(f1c[i]), 3),
                          'soporte': int(sup[i])} for i in range(len(clases))]
            # entrenar en todo para importancia y para guardar
            pipe_full = Pipeline([('sc', StandardScaler()), ('clf', fabrica())]).fit(Xv, y)
            importancia = _importancia(nombre, pipe_full, Xv, y, feature_names)
            rep = {
                'algoritmo': nombre,
                'f1_macro': round(f1m, 3), 'accuracy': round(acc, 3), 'balanced_accuracy': round(bacc, 3),
                'matriz_confusion': cm, 'clases': [str(c) for c in clases],
                'por_clase': por_clase, 'importancia': importancia,
            }
            reportes.append(rep)
            if mejor is None or f1m > mejor['_f1']:
                mejor = {'_f1': f1m, 'nombre': nombre, 'pipe': pipe_full, 'reporte': rep}
        except Exception as e:
            reportes.append({'algoritmo': nombre, 'error': str(e)})

    reportes.sort(key=lambda r: -(r.get('f1_macro') or -1))
    return {
        'mejor_algoritmo': mejor['nombre'] if mejor else None,
        'modelo': mejor['pipe'] if mejor else None,
        'comparacion': reportes,
    }

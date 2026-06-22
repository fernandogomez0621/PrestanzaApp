"""
core/pipeline.py
Orquesta todo el ML del sistema Prestanza, reutilizando el código ya validado:
  - data_processor: ETL + selección de titular como deudor principal
  - clasificador_riesgo: ingeniería de features + agrupación 3 clases
Entrena 4 familias de modelos por horizonte (6M, 12M):
  1) nueve_clases       -> calificación 0-10 (las 9 clases originales), todas las variables
  2) tres_clases        -> Buena/Media/Riesgo (clasificador ordinal)
  3) buena_no_buena     -> binario, todas las variables con ingeniería
  4) buena_no_buena_shap-> binario, SOLO variables con dirección SHAP correcta
Calcula SHAP, guarda cada entrenamiento en carpeta con timestamp y permite predecir.
"""
import os, json, glob, datetime, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(__file__))

from io_utils import leer_tabla, educacion_a_ordinal, EDUCATION_ORDINAL_FEATURE
from data_processor import DataProcessor
from clasificador_riesgo import ingenieria_features, agrupar_calificacion, OrdinalClassifier, ORDEN

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # backend/
VERSIONES_DIR = os.path.join(BASE_DIR, 'modelos_versionados')
os.makedirs(VERSIONES_DIR, exist_ok=True)

# Dirección esperada por lógica de crédito (para validar SHAP): +1 mayor->Buena, -1 mayor->No-buena
DIRECCION_ESPERADA = {
    'debtor_credit_score': +1, 'debtor_education_ordinal': +1, 'original_amount': -1,
    'public_db_no_lawsuits_defendant_money': -1, 'public_db_no_lawsuits': -1,
    'quotefactor': +1, 'value_monthly_payments': -1, 'snr_estimated_unique_properties': +1,
    'log_tax_report_total_income_value': +1, 'log_bank_monthly_avg_blance': +1,
    'collateral_commercial_value': +1, 'debtor_education_ordinal': +1, 'score_bajo': -1,
}


# --------------------------------------------------------------------------- #
# PROCESAMIENTO DE DATOS                                                       #
# --------------------------------------------------------------------------- #
def procesar_horizonte(datapoints_path, calificaciones_path, modo_seleccion='titular', pesos=None, reglas=None):
    """Devuelve el dataframe modelable (1 fila por crédito) para un horizonte.
    modo_seleccion: 'titular' o 'perfil'. pesos/reglas: configuración opcional del puntaje."""
    dp = DataProcessor()
    if pesos:
        dp.set_pesos(pesos)
    if reglas:
        dp.set_reglas(reglas)
    datos = dp.procesar_dataset_completo(datapoints_path, calificaciones_path, modo_seleccion=modo_seleccion)
    return datos


def features_modelo(datos):
    """Matriz X (con ingeniería) y target numérico y."""
    from model_trainer import ModelTrainer
    X, y = ModelTrainer(random_seed=42).preprocesar_datos(datos, usar_split=False)[:2]
    X = ingenieria_features(pd.DataFrame(X).reset_index(drop=True))
    y = pd.Series(y).reset_index(drop=True)
    return X, y


# --------------------------------------------------------------------------- #
# SHAP: qué variables tienen sentido                                          #
# --------------------------------------------------------------------------- #
def analizar_shap(X, y_binario):
    """Entrena un modelo lineal y devuelve, por variable, importancia SHAP y si la
    dirección coincide con la lógica de crédito esperada."""
    import shap
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(class_weight='balanced', max_iter=3000).fit(Xs, y_binario)
    sv = shap.LinearExplainer(clf, Xs).shap_values(Xs)
    filas = []
    for i, c in enumerate(X.columns):
        imp = float(np.abs(sv[:, i]).mean())
        corr = np.corrcoef(X[c], sv[:, i])[0, 1]
        signo = int(np.sign(corr)) if not np.isnan(corr) else 0
        esperado = DIRECCION_ESPERADA.get(c, None)
        tiene_sentido = (esperado is not None and signo == esperado)
        filas.append({
            'variable': c, 'importancia_shap': round(imp, 4),
            'direccion': 'mayor → Buena' if signo > 0 else 'mayor → No-buena',
            'esperado': (None if esperado is None else ('mayor → Buena' if esperado > 0 else 'mayor → No-buena')),
            'tiene_sentido': bool(tiene_sentido),
        })
    filas.sort(key=lambda d: -d['importancia_shap'])
    vars_con_sentido = [f['variable'] for f in filas if f['tiene_sentido']]
    return filas, vars_con_sentido


# --------------------------------------------------------------------------- #
# ENTRENAMIENTO DE LAS 4 FAMILIAS                                             #
# --------------------------------------------------------------------------- #
def _cv_f1(X, y, estimator):
    try:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        return float(np.mean(cross_val_score(estimator, X, y, cv=cv, scoring='f1_macro')))
    except Exception:
        return None


def entrenar_horizonte(datos, periodo, cortes=None):
    """Entrena las 4 familias para un horizonte comparando varios algoritmos y
    eligiendo el mejor por F1-macro. Devuelve un bundle serializable con la
    comparación, matrices de confusión, métricas por clase e importancia."""
    from multimodelo import comparar
    X, y = features_modelo(datos)
    feature_names = list(X.columns)
    clase3 = y.apply(lambda v: agrupar_calificacion(v, cortes))
    yB = (clase3 == 'Buena').astype(int).values

    bundle = {'periodo': periodo, 'feature_names': feature_names, 'n': int(len(X)),
              'modelos': {}, 'metricas': {}, 'shap': {}, 'comparacion': {}}

    # --- 1) NUEVE CLASES (calificación cruda) ---
    y9 = y.astype(int).astype(str).values
    orden9 = sorted(set(y9), key=lambda v: -int(v))
    r9 = comparar(X, y9, feature_names, orden_clases=orden9)
    bundle['modelos']['nueve_clases'] = r9['modelo']
    bundle['comparacion']['nueve_clases'] = r9['comparacion']
    bundle['metricas']['nueve_clases'] = {'mejor_algoritmo': r9['mejor_algoritmo'],
        'f1_macro_cv': r9['comparacion'][0].get('f1_macro') if r9['comparacion'] else None,
        'clases': orden9}

    # --- 2) TRES CLASES ---
    r3 = comparar(X, clase3.values, feature_names, orden_clases=['Buena', 'Media', 'Riesgo'])
    bundle['modelos']['tres_clases'] = r3['modelo']
    bundle['comparacion']['tres_clases'] = r3['comparacion']
    bundle['metricas']['tres_clases'] = {'mejor_algoritmo': r3['mejor_algoritmo'],
        'f1_macro_cv': r3['comparacion'][0].get('f1_macro') if r3['comparacion'] else None}

    # --- 3) BUENA vs NO-BUENA (todas las variables) ---
    rB = comparar(X, yB, feature_names, orden_clases=[0, 1])
    bundle['modelos']['buena_no_buena'] = rB['modelo']
    bundle['comparacion']['buena_no_buena'] = rB['comparacion']
    bundle['metricas']['buena_no_buena'] = _resumen_binario(rB)

    # --- SHAP + variables con sentido (sobre LogReg, para interpretabilidad) ---
    filas_shap, vars_sentido = analizar_shap(X, yB)
    bundle['shap']['buena_no_buena'] = filas_shap
    bundle['shap']['variables_con_sentido'] = vars_sentido

    # --- 4) BUENA vs NO-BUENA solo variables con sentido SHAP ---
    if len(vars_sentido) >= 2:
        Xs = X[vars_sentido]
        rBs = comparar(Xs, yB, vars_sentido, orden_clases=[0, 1])
        bundle['modelos']['buena_no_buena_shap'] = rBs['modelo']
        bundle['comparacion']['buena_no_buena_shap'] = rBs['comparacion']
        bundle['metricas']['buena_no_buena_shap'] = _resumen_binario(rBs)
        bundle['shap']['features_modelo_shap'] = vars_sentido
    else:
        bundle['modelos']['buena_no_buena_shap'] = None
        bundle['metricas']['buena_no_buena_shap'] = None

    return bundle


def _resumen_binario(r):
    """Métricas resumidas del mejor binario, compatibles con la UI previa."""
    if not r['comparacion']:
        return None
    mejor = r['comparacion'][0]
    pc = {d['clase']: d for d in mejor.get('por_clase', [])}
    buena = pc.get('1', {})
    return {
        'mejor_algoritmo': r['mejor_algoritmo'],
        'f1_macro_cv': mejor.get('f1_macro'),
        'accuracy': mejor.get('accuracy'),
        'balanced_accuracy': mejor.get('balanced_accuracy'),
        'precision_buena': buena.get('precision'),
        'recall_buena': buena.get('recall'),
    }


def _metricas_binarias(X, yB):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(class_weight='balanced', max_iter=3000))])
    proba = cross_val_predict(pipe, X, yB, cv=cv, method='predict_proba')[:, 1]
    pred = (proba >= 0.5).astype(int)
    dice_buena = pred.sum()
    return {
        'f1_macro_cv': round(float(f1_score(yB, pred, average='macro')), 3),
        'accuracy': round(float(accuracy_score(yB, pred)), 3),
        'balanced_accuracy': round(float(balanced_accuracy_score(yB, pred)), 3),
        'precision_buena': round(float(pred[yB == 1].sum() / dice_buena), 3) if dice_buena else None,
        'recall_buena': round(float(pred[yB == 1].sum() / max((yB == 1).sum(), 1)), 3),
    }


# --------------------------------------------------------------------------- #
# VERSIONADO                                                                   #
# --------------------------------------------------------------------------- #
def guardar_version(bundles, meta):
    """Guarda los bundles 6M/12M en una carpeta con timestamp."""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    carpeta = os.path.join(VERSIONES_DIR, ts)
    os.makedirs(carpeta, exist_ok=True)
    for periodo, bundle in bundles.items():
        if bundle is None:
            continue
        joblib.dump(bundle, os.path.join(carpeta, f'bundle_{periodo}.joblib'))
    meta_pub = {'version': ts, 'fecha': ts, **meta,
                'metricas': {p: (b['metricas'] if b else None) for p, b in bundles.items()}}
    with open(os.path.join(carpeta, 'meta.json'), 'w') as f:
        json.dump(meta_pub, f, indent=2, default=str)
    return ts


def listar_versiones():
    out = []
    for d in sorted(glob.glob(os.path.join(VERSIONES_DIR, '*')), reverse=True):
        mp = os.path.join(d, 'meta.json')
        if os.path.isdir(d) and os.path.exists(mp):
            out.append(json.load(open(mp)))
    return out


def cargar_version(version):
    carpeta = os.path.join(VERSIONES_DIR, version)
    bundles = {}
    for periodo in ['6M', '12M']:
        p = os.path.join(carpeta, f'bundle_{periodo}.joblib')
        bundles[periodo] = joblib.load(p) if os.path.exists(p) else None
    return bundles


# --------------------------------------------------------------------------- #
# PREDICCIÓN sobre un CSV de cliente (con codeudores)                          #
# --------------------------------------------------------------------------- #
def predecir_csv(datapoints_path, bundles, modo_seleccion='titular', pesos=None, reglas=None):
    """Procesa un CSV de datapoints y devuelve, por simulación, cada deudor con su
    predicción y el DESGLOSE del puntaje de perfil (según pesos y reglas)."""
    import re
    dp = DataProcessor()
    if pesos:
        dp.set_pesos(pesos)
    if reglas:
        dp.set_reglas(reglas)
    df = leer_tabla(datapoints_path)
    from io_utils import normalizar_datapoints
    df = normalizar_datapoints(df)
    if 'id_usuario (opcional)' not in df.columns and 'id_usuario' in df.columns:
        df['id_usuario (opcional)'] = df['id_usuario']

    nombres = dp.TARGET_DATA_NAMES + dp.META_DATA_NAMES
    piv = df[df['data_name'].isin(nombres)].pivot_table(
        index=['id_simulacion', 'id_usuario (opcional)'], columns='data_name',
        values='value', aggfunc='first').reset_index()
    piv.columns.name = None
    for c in dp.TARGET_DATA_NAMES:
        if c not in piv:
            piv[c] = np.nan
    piv = dp.estandarizar_datos(piv)
    piv = dp.calcular_variables_derivadas(piv)

    def tipo(v):
        m = re.search(r'[1-9]', str(v)); return int(m.group()) if m else None

    resultados = []
    for sim, grupo in piv.groupby('id_simulacion'):
        # determinar el deudor principal del grupo según el modo
        fila_ppal, _, _ = dp.seleccionar_deudor_principal(grupo, modo=modo_seleccion)
        id_ppal = str(fila_ppal.get('id_usuario (opcional)'))

        deudores = []
        for _, fila in grupo.iterrows():
            id_u = str(fila.get('id_usuario (opcional)'))
            es_ppal = (id_u == id_ppal)
            puntaje_total, desglose = dp.calcular_puntaje_total(fila)
            preds = {}
            Xf = _fila_a_features(fila)
            for periodo, bundle in bundles.items():
                if bundle is None:
                    continue
                preds[periodo] = _predecir_fila(Xf, bundle)
            deudores.append({
                'id_usuario': id_u,
                'es_deudor_principal': bool(es_ppal),
                'tipo': tipo(fila.get('id_type_debtor')),
                'credit_score': _num(fila.get('debtor_credit_score')),
                'educacion': str(fila.get('debtor_level_of_education')),
                'puntaje_total': round(float(puntaje_total), 4),
                'puntaje_desglose': _desglose_pesos(dp, desglose),
                'predicciones': preds,
            })
        deudores.sort(key=lambda d: (not d['es_deudor_principal'], -d['puntaje_total']))
        resultados.append({'id_simulacion': str(sim), 'modo_seleccion': modo_seleccion, 'deudores': deudores})
    return resultados


# mapeo entre las llaves del desglose y la llave del peso correspondiente
_MAP_DESGLOSE_PESO = {
    'educacion': 'debtor_level_of_education', 'equity': 'equityFactor', 'quote': 'quotefactor',
    'credit_score': 'debtor_credit_score', 'creditos_cerrados': 'creditos_cerrados_factor',
    'similar_loans': 'debtor_closed_no_similar_loans', 'bank_inflow': 'bank_inflow_factor',
    'bank_average': 'bank_average_factor', 'propiedades': 'snr_estimated_unique_properties',
    'demandas': 'public_db_no_lawsuits', 'demandas_dinero': 'public_db_no_lawsuits_defendant_money',
}


def _desglose_pesos(dp, desglose):
    """Construye el desglose factor por factor: puntaje base, peso aplicado y aporte."""
    filas = []
    for k, base in desglose.items():
        peso_key = _MAP_DESGLOSE_PESO.get(k)
        peso = dp.PESOS_SELECCION.get(peso_key) if peso_key else None
        if peso is None:
            continue
        # las demandas restan
        signo = -1 if 'demandas' in k else 1
        aporte = signo * peso * float(base)
        filas.append({
            'factor': k, 'puntaje_base': round(float(base), 3),
            'peso': round(float(peso), 3), 'aporte': round(float(aporte), 4),
        })
    filas.sort(key=lambda f: -abs(f['aporte']))
    return filas


def _num(v):
    try: return float(v)
    except Exception: return None


def _fila_a_features(fila):
    base = pd.DataFrame([fila.to_dict()])
    if EDUCATION_ORDINAL_FEATURE not in base.columns:
        base[EDUCATION_ORDINAL_FEATURE] = base.get('debtor_level_of_education', pd.Series(['Desconocido'])).apply(educacion_a_ordinal)
    X = ingenieria_features(base)
    return X


def _alinear(X, feats):
    X = X.copy()
    for f in feats:
        if f not in X.columns:
            X[f] = 0.0
    return X[feats].apply(pd.to_numeric, errors='coerce').fillna(0)


def _predecir_fila(Xf, bundle):
    out = {}
    feats = bundle['feature_names']
    Xa = _alinear(Xf, feats)
    # 9 clases
    try: out['nueve_clases'] = str(bundle['modelos']['nueve_clases'].predict(Xa)[0])
    except Exception: out['nueve_clases'] = None
    # 3 clases
    try: out['tres_clases'] = str(bundle['modelos']['tres_clases'].predict(Xa)[0])
    except Exception: out['tres_clases'] = None
    # buena vs no buena
    try:
        p = bundle['modelos']['buena_no_buena'].predict_proba(Xa)[0, 1]
        out['buena_no_buena'] = {'etiqueta': 'Buena' if p >= 0.5 else 'No-Buena', 'prob_buena': round(float(p), 3)}
    except Exception: out['buena_no_buena'] = None
    # buena vs no buena SHAP
    try:
        if bundle['modelos'].get('buena_no_buena_shap') is not None:
            fs = bundle['shap']['features_modelo_shap']
            Xs = _alinear(Xf, fs)
            p = bundle['modelos']['buena_no_buena_shap'].predict_proba(Xs)[0, 1]
            out['buena_no_buena_shap'] = {'etiqueta': 'Buena' if p >= 0.5 else 'No-Buena', 'prob_buena': round(float(p), 3)}
    except Exception: out['buena_no_buena_shap'] = None
    return out

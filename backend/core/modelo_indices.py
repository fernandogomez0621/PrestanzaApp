"""
core/modelo_indices.py — Modelos de riesgo por índices (6M y 12M). Implementación de referencia.

Lo usa main_indices.py (servicio del puerto 8502). Reutiliza data_processor.py, io_utils.py y clasificador_riesgo.py.

  construir_variables(datapoints)  -> (V12, V6): una fila por operación (id_simulacion)
  ModeloIndices('12M' | '6M').fit(V, valor_calificacion, valor_12m=None)
  ModeloIndices.predecir(V)        -> índices, puntaje, banda y decisión por operación
  ModeloIndices.guardar(ruta) / ModeloIndices.cargar(ruta)   (JSON, sin pickle)

Convención de signos: +1 = un valor más alto de la variable implica MÁS riesgo. Puntaje alto = más riesgo.
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor                      # noqa: E402
from io_utils import educacion_a_ordinal, normalizar_datapoints  # noqa: E402
from clasificador_riesgo import ingenieria_features            # noqa: E402

U = 'id_usuario (opcional)'
VERSION_ESPEC = '1.0'

# --------------------------------------------------------------------------------------------- definiciones
INDICES_12M = {
    'mora_titular': {'t_debtor_current_delinquency_value': 1, 't_debtor_total_no_bad_reports': 1, 't_debtor_default_loans': 1,
                     't_debtor_closed_no_loans_w_bad_reports': 1, 'r_mora_sobre_deuda': 1, 'r_bad_reports_ratio': 1},
    'mora_codeudores': {'co_debtor_current_delinquency_value_sum': 1, 'co_debtor_total_no_bad_reports_max': 1,
                        'co_debtor_default_loans_max': 1},
    'peor_deudor': {'h_n_con_mora': 1, 'h_default_max': 1, 'h_score_min': -1},
    'carga_titular': {'r_cuota_sobre_ingbureau': 1, 'r_cuota_sobre_credbanco': 1, 'r_tcr_uso': 1, 't_debtor_current_total_payment': 1},
    'carga_hogar': {'r_carga_total': 1, 'r_cuota_sobre_ingreso': 1, 'r_deuda_buro_sobre_ingreso': 1, 'co_debtor_current_total_payment_sum': 1},
    'respaldo': {'r_ltv': 1, 'r_patrimonio_sobre_monto': -1, 'h_props': -1, 'h_equity': -1, 'r_impuesto_predial_mora': 1,
                 'l_collateral_tax_delinquency_value': 1},
    'score_titular': {'t_debtor_credit_score': -1},
    'score_codeudor': {'co_debtor_credit_score_max': -1, 'co_debtor_credit_score_min': -1},
    'estabilidad': {'t_debtor_tenure_at_curr_address': -1, 't_debtor_duration_of_curr_employment': -1, 't_debtor_years_experience': -1,
                    't_edu': -1, 't_debtor_social_stratum': -1},
    'litigios': {'t_public_db_no_lawsuits': 1, 't_public_db_no_lawsuits_defendant': 1, 't_public_db_no_lawsuits_defendant_money': 1},
    'analista': {'l_analist_score': -1, 'l_advisor_score': -1},
}
INDICES_6M = {
    'capacidad_titular': {'quotefactor': -1, 'score_x_quote': -1, 'tax_report_total_income_value': -1},
    'tamaño_crédito': {'original_amount': 1, 'value_monthly_payments': 1, 'LTVfactor': 1},
    'score': {'debtor_credit_score': -1, 'score_bajo': 1, 'score_x_educacion': -1},
    'educación': {'debtor_education_ordinal': -1},
    'litigios': {'public_db_no_lawsuits': 1, 'public_db_no_lawsuits_defendant_money': 1},
    'analista': {'l_analist_score': -1, 'l_advisor_score': -1},
    'garantía_casa': {'garantia_casa': 1},
}
FACTOR_MARCADO = 1.5          # se marca como No-Buena el 1,5 x la tasa de No-Buena del entrenamiento
CORTE_BUENA = 9               # por defecto: Valor_calificacion >= 9 -> Buena (se puede cambiar en 'Letras & clases')


def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


# --------------------------------------------------------------------------------------------- variables
_NUM_PERSONA = [
    'debtor_social_stratum', 'debtor_tenure_at_curr_address', 'debtor_duration_of_curr_employment', 'debtor_years_experience',
    'debtor_closed_no_loans_w_bad_reports', 'debtor_credit_agency_min_probable_income', 'debtor_credit_score',
    'debtor_current_delinquency_value', 'debtor_current_loan_value', 'debtor_current_tcr_usage', 'debtor_current_total_payment',
    'debtor_default_loans', 'debtor_total_no_bad_reports', 'debtor_total_no_loans', 'debtor_total_no_loans_w_bad_reports',
    'bank_monthly_avg_credit_value', 'bank_monthly_avg_blance', 'tax_report_total_income_value', 'tax_report_total_equity_value',
    'snr_estimated_unique_properties', 'public_db_no_lawsuits', 'public_db_no_lawsuits_defendant',
    'public_db_no_lawsuits_defendant_money']
_NUM_CREDITO = ['original_amount', 'value_monthly_payments', 'collateral_commercial_value', 'collateral_tax_delinquency_value',
                'analist_score', 'advisor_score']
_CUENTAS_CERO = ['t_public_db_no_lawsuits', 't_public_db_no_lawsuits_defendant', 't_public_db_no_lawsuits_defendant_money']


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def variables_12m(raw, dp=None):
    """Variables por operación: titular (t_), crédito (l_), codeudores (co_), hogar (h_) y razones (r_). Faltantes = NaN."""
    dp = dp or DataProcessor()
    raw = raw.copy(); raw['_f'] = pd.to_datetime(raw.fecha_creacion, dayfirst=True, errors='coerce')
    raw = raw[raw['_f'] == raw.groupby(['id_simulacion', U])['_f'].transform('max')]       # último registro por persona
    pv = raw.pivot_table(index=['id_simulacion', U], columns='data_name', values='value', aggfunc='first').reset_index()

    def num(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
        s = str(v).strip()
        if s in ('', 'is_missing', 'manual_review', 'nan') or '####' in s: return np.nan
        return dp.procesar_valor_numerico(s, permitir_negativos=True)
    for c in _NUM_PERSONA + _NUM_CREDITO + ['debtor_level_of_education', 'id_type_debtor', 'type_collateral']:
        if c not in pv: pv[c] = np.nan
    for c in _NUM_PERSONA + _NUM_CREDITO: pv[c] = pv[c].map(num)
    pv['edu'] = pv['debtor_level_of_education'].map(dp.procesar_debtor_level_of_education).map(educacion_a_ordinal)
    pv['tipo'] = pv['id_type_debtor'].map(dp._limpiar_tipo_deudor)
    pv.loc[pv['debtor_credit_score'] < 0, 'debtor_credit_score'] = np.nan                    # score negativo = empresa
    pv['tiene_loan'] = pv['original_amount'].notna()
    pv['rank'] = np.where(pv.tipo == 1, 0, np.where(pv.tiene_loan, 1, 2))
    pv = pv.sort_values(['id_simulacion', 'rank'])
    tit = pv.groupby('id_simulacion').head(1).set_index('id_simulacion')
    loan = pv[pv.tiene_loan].groupby('id_simulacion').head(1).set_index('id_simulacion')
    co = pv[~pv.index.isin(pv.groupby('id_simulacion').head(1).index)]
    ops = sorted(tit.index); X = pd.DataFrame(index=pd.Index(ops, name='id_simulacion'))
    for c in _NUM_PERSONA + ['edu']: X['t_' + c] = tit.loc[ops, c].astype(float)
    for c in _NUM_CREDITO: X['l_' + c] = loan[c].reindex(ops).astype(float)
    X['l_type_collateral'] = loan['type_collateral'].reindex(ops).fillna('na').astype(str).str.strip().str.lower()
    g = co.groupby('id_simulacion')
    for c, f in [('debtor_credit_score', 'max'), ('debtor_credit_score', 'min'), ('debtor_current_total_payment', 'sum'),
                 ('debtor_default_loans', 'max'), ('debtor_total_no_bad_reports', 'max'), ('debtor_current_delinquency_value', 'sum')]:
        X[f'co_{c}_{f}'] = g[c].agg(f).reindex(ops)
    allp = pv.groupby('id_simulacion')
    hs = lambda c: allp[c].sum(min_count=1).reindex(ops)
    X['h_income_tax_m'] = hs('tax_report_total_income_value') / 12
    X['h_income_bureau'] = hs('debtor_credit_agency_min_probable_income')
    X['h_bank_credit'] = hs('bank_monthly_avg_credit_value')
    X['h_payment_bureau'] = hs('debtor_current_total_payment') * 1000          # buró reporta en miles
    X['h_debt_bureau'] = hs('debtor_current_loan_value') * 1000
    X['h_equity'] = hs('tax_report_total_equity_value')
    X['h_props'] = hs('snr_estimated_unique_properties')
    X['h_score_min'] = allp['debtor_credit_score'].min().reindex(ops)
    X['h_default_max'] = allp['debtor_default_loans'].max().reindex(ops)
    X['h_n_con_mora'] = allp['debtor_current_delinquency_value'].apply(lambda s: (s > 0).sum()).reindex(ops)
    cuota = X['l_value_monthly_payments']
    ing = np.fmax(np.fmax(X['h_income_tax_m'].fillna(0), X['h_income_bureau'].fillna(0)), X['h_bank_credit'].fillna(0)).replace(0, np.nan)
    X['r_cuota_sobre_ingreso'] = _div(cuota, ing)
    X['r_carga_total'] = _div(cuota + X['h_payment_bureau'].fillna(0), ing)
    X['r_cuota_sobre_credbanco'] = _div(cuota, X['h_bank_credit'])
    X['r_cuota_sobre_ingbureau'] = _div(cuota, X['h_income_bureau'])
    X['r_ltv'] = _div(X['l_original_amount'], X['l_collateral_commercial_value'])
    X['r_patrimonio_sobre_monto'] = _div(X['h_equity'], X['l_original_amount'])
    X['r_deuda_buro_sobre_ingreso'] = _div(X['h_debt_bureau'], ing * 12)
    X['r_mora_sobre_deuda'] = _div(X['t_debtor_current_delinquency_value'], X['t_debtor_current_loan_value'])
    X['r_bad_reports_ratio'] = _div(X['t_debtor_total_no_loans_w_bad_reports'], X['t_debtor_total_no_loans'])
    X['r_tcr_uso'] = X['t_debtor_current_tcr_usage']
    X['r_impuesto_predial_mora'] = (X['l_collateral_tax_delinquency_value'] > 0).astype(float)
    for c in _CUENTAS_CERO: X[c] = X[c].fillna(0)                                            # sin registro = 0 demandas
    return X


def variables_6m(raw, dp=None):
    """Variables de la app (mismo ETL: titular, filtro de calidad, factores, ingeniería) + analista + garantía casa."""
    dp = dp or DataProcessor()
    piv = dp.transformar_datapoints(raw); piv['id_simulacion'] = piv['id_simulacion'].astype(str)
    d = dp.estandarizar_datos(piv.copy()); d = dp.aplicar_filtro_calidad(d)
    d = dp.calcular_variables_derivadas(d); d = dp.filtrar_duplicados(d)
    filas = []
    for sid, g in d.groupby('id_simulacion'):
        f, _, _ = dp.seleccionar_deudor_principal(g, modo='titular'); filas.append(f.to_dict())
    T = pd.DataFrame(filas)
    if 'debtor_education_ordinal' not in T and 'debtor_level_of_education' in T:
        T['debtor_education_ordinal'] = T['debtor_level_of_education'].apply(educacion_a_ordinal)
    T = ingenieria_features(T.select_dtypes('number').assign(id_simulacion=T['id_simulacion'].values)
                            .set_index('id_simulacion'))
    T.index = T.index.astype(str)
    return T


def construir_variables(datapoints):
    """datapoints: DataFrame en formato largo (como datapoints.csv). Devuelve (V12, V6) indexados por id_simulacion."""
    raw = normalizar_datapoints(datapoints.copy()); raw['id_simulacion'] = raw['id_simulacion'].astype(str); raw[U] = raw[U].astype(str)
    dp = DataProcessor(); V12 = variables_12m(raw, dp); V6 = variables_6m(raw, dp)
    V6 = V6.join(V12[['l_analist_score', 'l_advisor_score']], how='left')
    V6['garantia_casa'] = V12['l_type_collateral'].reindex(V6.index).eq('casa').astype(float)
    return V12, V6


# --------------------------------------------------------------------------------------------- modelo
class ModeloIndices:
    """Puntaje = suma de índices; índice = promedio con signo de variables estandarizadas con parámetros del entrenamiento.
    12M: log con signo + z-score (media/desv.), faltantes se omiten del promedio.
    6M : imputación por mediana + escalado robusto (mediana/IQR)."""

    def __init__(self, horizonte, corte=CORTE_BUENA):
        assert horizonte in ('6M', '12M'); self.h = horizonte; self.corte = float(corte)
        self.defs = INDICES_12M if horizonte == '12M' else INDICES_6M

    # -- estandarización
    def _z(self, V):
        A = V.reindex(columns=self.cols).astype(float)
        if self.h == '12M':
            return (slog(A) - pd.Series(self.p['mu'])) / pd.Series(self.p['sd'])
        A = A.fillna(pd.Series(self.p['mediana']))
        return (A - pd.Series(self.p['centro'])) / pd.Series(self.p['escala'])

    def fit(self, V, valor_calificacion, valor_12m=None):
        """V: variables (una fila por operación). valor_calificacion: Valor_calificacion del horizonte (NaN = sin calificar).
        6M: si se pasa valor_12m, se excluyen del ajuste los 'transitorios' (No-Buena a 6M y Buena a 12M)."""
        y = pd.Series(valor_calificacion, index=V.index).astype(float)
        m = y.notna()
        if self.h == '6M' and valor_12m is not None:
            y12 = pd.Series(valor_12m, index=V.index).astype(float)
            m &= ~((y < self.corte) & (y12 >= self.corte))
        Vt = V.loc[m]; yt = (y[m] < self.corte).astype(int)
        self.cols = sorted({c for v in self.defs.values() for c in v})
        faltan = [c for c in self.cols if c not in Vt]
        if faltan: raise ValueError(f'faltan variables: {faltan}')
        A = Vt[self.cols].astype(float)
        if self.h == '12M':
            L = slog(A); sd = L.std().replace(0, 1).fillna(1)
            self.p = {'mu': L.mean().fillna(0).to_dict(), 'sd': sd.to_dict()}
        else:
            med = A.median().fillna(0); B = A.fillna(med)
            q1, q2, q3 = B.quantile(.25), B.quantile(.5), B.quantile(.75); iqr = (q3 - q1).replace(0, 1)
            self.p = {'mediana': med.to_dict(), 'centro': q2.to_dict(), 'escala': iqr.to_dict()}
        s = self.puntaje(Vt)
        self.tasa_entrenamiento = float(yt.mean())
        self.frac_marcado = float(min(.95, FACTOR_MARCADO * self.tasa_entrenamiento))
        self.umbral = float(np.quantile(s, 1 - self.frac_marcado))
        self.terciles = [float(x) for x in np.quantile(s, [1 / 3, 2 / 3])]
        self.n_entrenamiento, self.malos_entrenamiento = int(len(yt)), int(yt.sum())
        self.fecha = datetime.datetime.now().isoformat(timespec='seconds')
        return self

    def indices(self, V):
        Z = self._z(V)
        return pd.DataFrame({k: pd.concat([Z[c] * s for c, s in v.items()], axis=1).mean(axis=1) for k, v in self.defs.items()},
                            index=V.index).fillna(0)

    def puntaje(self, V):
        return self.indices(V).sum(axis=1)

    def predecir(self, V):
        I = self.indices(V); s = I.sum(axis=1)
        out = I.round(4).copy(); out['puntaje'] = s.round(4)
        out['banda'] = np.where(s >= self.terciles[1], 'Alto', np.where(s >= self.terciles[0], 'Medio', 'Bajo'))
        out['decision'] = np.where(s >= self.umbral, 'No-Buena (revisar)', 'Buena')
        return out

    def guardar(self, ruta):
        json.dump(dict(version_especificacion=VERSION_ESPEC, horizonte=self.h, corte_buena=self.corte, definiciones=self.defs, columnas=self.cols, parametros=self.p,
                       umbral=self.umbral, terciles=self.terciles, frac_marcado=self.frac_marcado, tasa_entrenamiento=self.tasa_entrenamiento,
                       n_entrenamiento=self.n_entrenamiento, malos_entrenamiento=self.malos_entrenamiento, fecha=self.fecha),
                  open(ruta, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    @classmethod
    def cargar(cls, ruta):
        d = json.load(open(ruta, encoding='utf-8')); m = cls(d['horizonte'], d.get('corte_buena', CORTE_BUENA))
        m.defs, m.cols, m.p, m.umbral, m.terciles = d['definiciones'], d['columnas'], d['parametros'], d['umbral'], d['terciles']
        m.frac_marcado, m.tasa_entrenamiento = d['frac_marcado'], d['tasa_entrenamiento']
        m.n_entrenamiento, m.malos_entrenamiento, m.fecha = d['n_entrenamiento'], d['malos_entrenamiento'], d['fecha']
        return m


# --------------------------------------------------------------------------------------------- evaluación
def evaluar(h, V, valor_calificacion, valor_12m=None, repeticiones=10, semilla=0, corte=CORTE_BUENA):
    """Validación cruzada estratificada 5 x repeticiones con el mismo procedimiento del entrenamiento.
    6M: los transitorios se excluyen del ajuste pero se incluyen en la evaluación."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
    y = pd.Series(valor_calificacion, index=V.index).astype(float)
    m = y.notna(); Vm, ym = V.loc[m], y[m]
    yb = (ym < corte).astype(int).values                             # 1 = No-Buena
    y12 = pd.Series(valor_12m, index=V.index).astype(float).loc[m] if (h == '6M' and valor_12m is not None) else None
    defs = INDICES_12M if h == '12M' else INDICES_6M
    R, cm, por, bandas = [], np.zeros((2, 2), int), {k: [] for k in defs}, np.zeros((3, 2))
    for r in range(semilla, semilla + repeticiones):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Vm, yb):
            mod = ModeloIndices(h, corte).fit(Vm.iloc[tr], ym.iloc[tr], None if y12 is None else y12.iloc[tr])
            I = mod.indices(Vm.iloc[te]); s = I.sum(axis=1).values; q = (s >= mod.umbral).astype(int); yt = yb[te]
            R.append([roc_auc_score(yt, s), f1_score(yt, q, average='macro'), balanced_accuracy_score(yt, q)])
            cm += confusion_matrix(yt, q, labels=[0, 1])
            for k in defs:
                por[k].append(roc_auc_score(yt, I[k].values) if I[k].nunique() > 1 else .5)
            b = np.where(s >= mod.terciles[1], 2, np.where(s >= mod.terciles[0], 1, 0))
            for j in range(3): bandas[j] += [(b == j).sum(), yt[b == j].sum()]
    R = np.array(R); p, rc, f, sup = precision_recall_fscore_support(
        np.repeat([0, 0, 1, 1], cm.ravel()), np.repeat([0, 1, 0, 1], cm.ravel()), labels=[0, 1], zero_division=0)
    clases = ['Buena', 'No-Buena']
    return {
        'n': int(len(yb)), 'no_buena': int(yb.sum()), 'repeticiones': repeticiones, 'corte_buena': float(corte),
        'auc': round(float(R[:, 0].mean()), 3), 'auc_sd': round(float(R[:, 0].reshape(-1, 5).mean(1).std()), 3),
        'f1_macro': round(float(R[:, 1].mean()), 3), 'balanced_accuracy': round(float(R[:, 2].mean()), 3),
        'clases': clases, 'matriz_confusion': (cm / repeticiones).round(1).tolist(),
        'por_clase': [{'clase': c, 'precision': round(float(p[i]), 3), 'recall': round(float(rc[i]), 3), 'f1': round(float(f[i]), 3),
                       'soporte': int(round(sup[i] / repeticiones))} for i, c in enumerate(clases)],
        'por_indice': sorted([{'indice': k, 'auc_solo': round(float(np.mean(v)), 3)} for k, v in por.items()], key=lambda d: -d['auc_solo']),
        'bandas': [{'banda': nb, 'n_promedio': round(float(c[0] / repeticiones), 1), 'tasa_no_buena': round(float(c[1] / c[0]), 3) if c[0] else None}
                   for nb, c in zip(['Bajo', 'Medio', 'Alto'], bandas)],
    }

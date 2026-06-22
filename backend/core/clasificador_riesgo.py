"""
clasificador_riesgo.py
----------------------
Clasificador de riesgo crediticio en 3 clases ordinales: Riesgo < Media < Buena.

Mejoras frente al enfoque de regresión anterior:
  - Trata el problema como clasificación ordinal (descomposición de Frank & Hall):
    dos modelos binarios (P(>Riesgo) y P(>Buena)) que se combinan en UNA salida
    de 3 clases. Cada corte binario tiene mejor balance de datos y se aprende
    mejor que la multiclase directa.
  - Regresión logística con class_weight='balanced' (no ignora las clases chicas).
  - Variables generadas (logs de montos, interacciones, flags) que sí aportan.

Uso:
    from clasificador_riesgo import ClasificadorRiesgo
    c = ClasificadorRiesgo()
    c.entrenar(datos_df)            # datos_df = salida de DataProcessor (incluye Valor_calificacion)
    c.guardar('modelos_exportados/6_meses/clasificador_6M.joblib')
    # ...
    c2 = ClasificadorRiesgo.cargar('modelos_exportados/6_meses/clasificador_6M.joblib')
    etiqueta, probabilidades = c2.predecir(features_df)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ORDEN = ['Riesgo', 'Media', 'Buena']   # orden ascendente de calidad


def agrupar_calificacion(valor, cortes=None):
    """Convierte la calificación numérica a clase ordinal (Buena/Media/Riesgo).
    cortes: dict opcional {'Buena':9,'Media':6,'Riesgo':0} con umbrales mínimos."""
    if cortes:
        from calif_config import agrupar_con_cortes
        return agrupar_con_cortes(valor, cortes)
    v = float(valor)
    if v >= 9:
        return 'Buena'      # AAA, AA
    if v >= 6:
        return 'Media'      # A, BBB
    return 'Riesgo'        # BB y abajo


def ingenieria_features(X):
    """Genera variables derivadas e interacciones con sentido crediticio."""
    X = X.copy()
    eps = 1e-6
    for c in ['original_amount', 'collateral_commercial_value',
              'tax_report_total_income_value', 'tax_report_total_equity_value',
              'bank_monthly_avg_blance', 'bank_monthly_avg_credit_value',
              'value_monthly_payments', 'debtor_closed_loan_value']:
        if c in X.columns:
            X['log_' + c] = np.log1p(np.clip(X[c].astype(float), 0, None))
    if 'debtor_credit_score' in X and 'debtor_education_ordinal' in X:
        X['score_x_educacion'] = X['debtor_credit_score'] * X['debtor_education_ordinal']
    if 'quotefactor' in X and 'debtor_credit_score' in X:
        X['score_x_quote'] = X['debtor_credit_score'] * X['quotefactor']
    if 'equityFactor' in X and 'LTVfactor' in X:
        X['equity_sobre_ltv'] = X['equityFactor'] / (X['LTVfactor'] + eps)
    if 'bank_inflow_factor' in X and 'bank_average_factor' in X:
        X['solvencia_bancaria'] = X['bank_inflow_factor'] + X['bank_average_factor']
    if 'public_db_no_lawsuits_defendant_money' in X:
        X['flag_demanda_dinero'] = (X['public_db_no_lawsuits_defendant_money'].astype(float) > 0).astype(int)
    if 'public_db_no_lawsuits' in X:
        X['flag_demandas'] = (X['public_db_no_lawsuits'].astype(float) > 0).astype(int)
    if 'snr_estimated_unique_properties' in X:
        X['flag_tiene_propiedades'] = (X['snr_estimated_unique_properties'].astype(float) > 0).astype(int)
    if 'debtor_closed_no_similar_loans' in X:
        X['flag_historial_similar'] = (X['debtor_closed_no_similar_loans'].astype(float) > 0).astype(int)
    if 'debtor_credit_score' in X:
        X['score_alto'] = (X['debtor_credit_score'].astype(float) >= 700).astype(int)
        X['score_bajo'] = (X['debtor_credit_score'].astype(float) < 550).astype(int)
    return X.replace([np.inf, -np.inf], 0).fillna(0)


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Clasificación ordinal por descomposición binaria (Frank & Hall)."""
    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.clfs_ = []
        yidx = np.array([ORDEN.index(v) for v in y])
        for k in range(len(ORDEN) - 1):
            yk = (yidx > k).astype(int)
            self.clfs_.append(clone(self.base).fit(X, yk))
        self.classes_ = np.array(ORDEN)
        return self

    def predict_proba(self, X):
        g = [c.predict_proba(X)[:, 1] for c in self.clfs_]   # P(>Riesgo), P(>Media)
        pR = np.clip(1 - g[0], 0, None)
        pM = np.clip(g[0] - g[1], 0, None)
        pB = np.clip(g[1], 0, None)
        P = np.vstack([pR, pM, pB]).T
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class ClasificadorRiesgo:
    """Envoltura entrenable/serializable para el clasificador de 3 clases."""

    def __init__(self):
        self.pipeline = None
        self.feature_names = None

    def _preparar(self, datos_df):
        # Importación diferida para no acoplar al pipeline de entrenamiento
        from model_trainer import ModelTrainer
        X, y = ModelTrainer(random_seed=42).preprocesar_datos(datos_df, usar_split=False)[:2]
        X = pd.DataFrame(X).reset_index(drop=True)
        X = ingenieria_features(X)
        return X, y

    def entrenar(self, datos_df):
        X, y = self._preparar(datos_df)
        yc = pd.Series(y).apply(agrupar_calificacion).values
        self.feature_names = list(X.columns)
        self.pipeline = Pipeline([
            ('sc', StandardScaler()),
            ('ord', OrdinalClassifier(LogisticRegression(class_weight='balanced', max_iter=3000)))
        ]).fit(X, yc)
        return self

    def _alinear(self, features_df):
        """Aplica ingeniería y alinea columnas al esquema de entrenamiento."""
        X = ingenieria_features(pd.DataFrame(features_df).copy())
        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0.0
        return X[self.feature_names]

    def predecir(self, features_df):
        X = self._alinear(features_df)
        etiquetas = self.pipeline.predict(X)
        probas = self.pipeline.predict_proba(X)
        return etiquetas, pd.DataFrame(probas, columns=ORDEN, index=X.index)

    def guardar(self, ruta):
        joblib.dump({'pipeline': self.pipeline, 'feature_names': self.feature_names}, ruta)

    @classmethod
    def cargar(cls, ruta):
        d = joblib.load(ruta)
        obj = cls()
        obj.pipeline = d['pipeline']
        obj.feature_names = d['feature_names']
        return obj

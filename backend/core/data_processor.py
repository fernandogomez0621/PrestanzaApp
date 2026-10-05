import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import warnings

from io_utils import (
    leer_tabla,
    normalizar_datapoints,
    parsear_fechas,
    educacion_a_ordinal,
    EDUCATION_ORDINAL_FEATURE,
)

warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.mapeo_numeros_a_letras = {
            10: 'AAA', 9: 'AA', 8: 'A',
            6: 'BBB', 5: 'BB', 4: 'B',
            2: 'CCC', 1: 'CC', 0: 'C'
        }

        self.TARGET_DATA_NAMES = [
            "debtor_level_of_education",
            "original_amount",
            "tax_report_total_equity_value",
            "tax_report_total_income_value",
            "value_monthly_payments",
            "debtor_credit_score",
            "collateral_commercial_value",
            "public_db_no_lawsuits",
            "public_db_no_lawsuits_defendant_money",
            "bank_monthly_avg_credit_value",
            "bank_monthly_avg_blance",
            "debtor_closed_loan_value",
            "debtor_closed_no_similar_loans",
            "snr_estimated_unique_properties"
        ]

        # data_names que NO son features del modelo, pero que necesitamos
        # arrastrar por el pivote para identificar al TITULAR del crédito.
        # La calificación pertenece a la operación (id_loan_request), cuyo
        # dueño es el deudor con id_type_debtor == 1.
        self.META_DATA_NAMES = [
            "id_type_debtor",
            "id_loan_request",
        ]

        self.columnas_excluir = [
            'education (Educación)', 'IdOperacion', 'IdOperacion_x', 'IdOperacion_y',
            'fecha_creacion', 'Fecha_calificación (meses)_x', 'Fecha_calificación (meses)_y',
            'Tiene data-points'
        ]

        # Pesos para la selección del deudor principal (suman 100%)
        self.PESOS_SELECCION = {
            'debtor_level_of_education': 0.05,              # 5%
            'equityFactor': 0.08,                          # 8%
            'quotefactor': 0.12,                           # 12%
            'debtor_credit_score': 0.04,                   # 4%
            'creditos_cerrados_factor': 0.12,              # 12%
            'debtor_closed_no_similar_loans': 0.04,        # 4%
            'bank_inflow_factor': 0.12,                    # 12%
            'bank_average_factor': 0.08,                   # 8%
            'snr_estimated_unique_properties': 0.10,       # 10%
            'public_db_no_lawsuits': 0.05,                 # 5% (negativo)
            'public_db_no_lawsuits_defendant_money': 0.20  # 20% (negativo)
        }
        # pesos por defecto, para poder restaurar
        self.PESOS_DEFAULT = dict(self.PESOS_SELECCION)
        # Reglas de puntaje por escalones (configurables)
        import copy as _copy
        from reglas_puntaje import CONFIG_DEFAULT as _RCFG
        self.REGLAS = _copy.deepcopy(_RCFG)

    def set_pesos(self, pesos):
        """Sobrescribe los pesos de selección (los que no vengan se dejan igual)."""
        if pesos:
            for k, v in pesos.items():
                if k in self.PESOS_SELECCION:
                    self.PESOS_SELECCION[k] = float(v)
        return self.PESOS_SELECCION

    def set_reglas(self, reglas):
        """Sobrescribe la configuración de escalones/categorías del puntaje."""
        if reglas:
            import copy
            self.REGLAS = copy.deepcopy(reglas)
        return self.REGLAS

    def calcular_puntaje_educacion(self, educacion):
        """
        Calcula puntaje de educación según las reglas:
        0 puntos -> sin estudio, primaria, bachillerato
        5 puntos -> técnico profesional, tecnológico
        8 puntos -> pregrado
        10 puntos -> especialización, maestría, doctorado
        """
        mapeo_educacion = {
            'sin estudio': 0,
            'primaria': 0,
            'bachillerato': 0,
            'técnico profesional': 5,
            'tecnológico': 5,
            'pregrado': 8,
            'especialización': 10,
            'maestría': 10,
            'doctorado': 10,
            'Desconocido': 0
        }
        return mapeo_educacion.get(educacion, 0)

    def calcular_puntaje_equity_factor(self, equity_factor):
        """
        Calcula puntaje del equity factor:
        0 puntos -> equityFactor <= 2
        X puntos -> equityFactor > 2 And equityFactor <= 10 (interpolación lineal)
        10 puntos -> equityFactor > 10
        """
        if equity_factor <= 2:
            return 0
        elif equity_factor > 10:
            return 10
        else:
            # Interpolación lineal entre 2 y 10
            return ((equity_factor - 2) / 8) * 10

    def calcular_puntaje_quote_factor(self, quote_factor):
        """
        Calcula puntaje del quote factor (income factor):
        0 puntos -> incomeFactor < 5
        6 puntos -> incomeFactor >= 5 And incomeFactor < 10
        8 puntos -> incomeFactor >= 10 And incomeFactor < 20
        10 puntos -> incomeFactor >= 20
        """
        if quote_factor < 5:
            return 0
        elif quote_factor < 10:
            return 6
        elif quote_factor < 20:
            return 8
        else:
            return 10

    def calcular_puntaje_credit_score(self, credit_score):
        """
        Calcula puntaje del credit score:
        0 puntos -> creditScore < 450
        1 puntos -> creditScore >= 450 And creditScore < 550
        x puntos -> creditScore >= 550 And creditScore < 800 (fórmula lineal)
        11 puntos -> creditScore >= 800 And creditScore < 900
        12 puntos -> creditScore >= 900
        """
        if credit_score < 450:
            return 0
        elif credit_score < 550:
            return 1
        elif credit_score < 800:
            # Interpolación lineal entre 550 y 800: de 1 a 11 puntos
            return 1 + ((credit_score - 550) / 250) * 10
        elif credit_score < 900:
            return 11
        else:
            return 12

    def calcular_puntaje_creditos_cerrados(self, creditos_cerrados_factor):
        """
        Calcula puntaje de créditos cerrados:
        0-9 puntos dependiendo del factor
        10 puntos si el factor es mayor a 10
        """
        if creditos_cerrados_factor <= 0:
            return 0
        elif creditos_cerrados_factor >= 10:
            return 10
        else:
            return min(9, creditos_cerrados_factor)

    def calcular_puntaje_similar_loans(self, similar_loans):
        """
        Calcula puntaje de créditos similares
        Entre más créditos similares, mejor puntaje (máximo 10)
        """
        return min(10, similar_loans * 2)

    def calcular_puntaje_bank_factors(self, bank_factor):
        """
        Calcula puntaje de factores bancarios
        Entre más alto el factor, mejor puntaje (máximo 12)
        """
        if bank_factor <= 0:
            return 0
        elif bank_factor >= 10:
            return 12
        else:
            return min(12, bank_factor * 1.2)

    def calcular_puntaje_propiedades(self, propiedades):
        """
        Calcula puntaje de propiedades
        Entre más propiedades, mejor puntaje (máximo 10)
        """
        return min(10, propiedades * 2)

    def calcular_puntaje_demandas_generales(self, demandas):
        """
        Calcula puntaje de demandas generales (negativo):
        -1 punto por demandas no relacionadas con plata
        -0.5 puntos por cada demanda puesta
        """
        return -min(10, demandas / 3)  # Dividir por 3 como indica la fórmula

    def calcular_puntaje_demandas_dinero(self, demandas_dinero):
        """
        Calcula puntaje de demandas por dinero (negativo):
        -3 puntos por demanda ejecutiva
        """
        return -min(15, demandas_dinero / 2)  # Dividir por 2 como indica la fórmula

    def calcular_puntaje_total(self, fila):
        """Calcula el puntaje total usando la config de reglas por escalones (editable)."""
        from reglas_puntaje import evaluar_escalon, evaluar_categoria
        esc = self.REGLAS.get('escalones', {})
        cat = self.REGLAS.get('categorias', {})

        puntajes = {}
        puntajes['educacion'] = evaluar_categoria(fila.get('debtor_level_of_education'), cat.get('educacion', {}))
        puntajes['equity'] = evaluar_escalon(fila.get('equityFactor'), esc.get('equity', []))
        puntajes['quote'] = evaluar_escalon(fila.get('quotefactor'), esc.get('quote', []))
        puntajes['credit_score'] = evaluar_escalon(fila.get('debtor_credit_score'), esc.get('credit_score', []))
        puntajes['creditos_cerrados'] = evaluar_escalon(fila.get('creditos_cerrados_factor'), esc.get('creditos_cerrados', []))
        puntajes['similar_loans'] = evaluar_escalon(fila.get('debtor_closed_no_similar_loans'), esc.get('similar_loans', []))
        puntajes['bank_inflow'] = evaluar_escalon(fila.get('bank_inflow_factor'), esc.get('bank', []))
        puntajes['bank_average'] = evaluar_escalon(fila.get('bank_average_factor'), esc.get('bank', []))
        puntajes['propiedades'] = evaluar_escalon(fila.get('snr_estimated_unique_properties'), esc.get('propiedades', []))
        puntajes['demandas'] = evaluar_escalon(fila.get('public_db_no_lawsuits'), esc.get('demandas', []))
        puntajes['demandas_dinero'] = evaluar_escalon(fila.get('public_db_no_lawsuits_defendant_money'), esc.get('demandas_dinero', []))

        P = self.PESOS_SELECCION
        puntaje_total = (
            P['debtor_level_of_education'] * puntajes['educacion'] +
            P['equityFactor'] * puntajes['equity'] +
            P['quotefactor'] * puntajes['quote'] +
            P['debtor_credit_score'] * puntajes['credit_score'] +
            P['creditos_cerrados_factor'] * puntajes['creditos_cerrados'] +
            P['debtor_closed_no_similar_loans'] * puntajes['similar_loans'] +
            P['bank_inflow_factor'] * puntajes['bank_inflow'] +
            P['bank_average_factor'] * puntajes['bank_average'] +
            P['snr_estimated_unique_properties'] * puntajes['propiedades'] -
            P['public_db_no_lawsuits'] * puntajes['demandas'] -
            P['public_db_no_lawsuits_defendant_money'] * puntajes['demandas_dinero']
        )
        return puntaje_total, puntajes

    def _limpiar_tipo_deudor(self, valor):
        """
        Limpia id_type_debtor que puede venir corrupto desde Excel
        ('$ 1,00', '1/01/1900 0:00', etc.). Devuelve 1, 2, ... o None.
        """
        if pd.isna(valor):
            return None
        s = str(valor)
        m = re.search(r'[1-9]', s)
        return int(m.group()) if m else None

    def seleccionar_deudor_principal(self, grupo_df, modo='titular'):
        """
        Selecciona el deudor principal del grupo según el modo elegido:

        modo='titular'   -> usa el TITULAR del crédito (id_type_debtor == 1), dueño
                            de la operación a la que pertenece la calificación.
                            Si no hay tipo identificable, desempata por puntaje.
        modo='perfil'    -> REGLA DE NEGOCIO: elige al deudor (titular o codeudor)
                            con el MEJOR PUNTAJE de perfil, según los pesos configurados.

        Devuelve (fila_elegida, puntaje_total, puntajes_detalle).
        """
        if len(grupo_df) == 1:
            fila = grupo_df.iloc[0]
            puntaje_total, puntajes_detalle = self.calcular_puntaje_total(fila)
            return fila, puntaje_total, puntajes_detalle

        if modo == 'perfil':
            # Todos los deudores compiten por puntaje (regla de negocio)
            candidatos = grupo_df
        else:
            # modo titular: quedarnos con los tipo == 1
            if 'id_type_debtor' in grupo_df.columns:
                tipos = grupo_df['id_type_debtor'].apply(self._limpiar_tipo_deudor)
                titulares = grupo_df[tipos == 1]
            else:
                titulares = grupo_df.iloc[0:0]
            candidatos = titulares if len(titulares) >= 1 else grupo_df

        resultados = []
        for _, fila in candidatos.iterrows():
            puntaje_total, puntajes_detalle = self.calcular_puntaje_total(fila)
            resultados.append((fila, puntaje_total, puntajes_detalle))

        mejor_resultado = max(resultados, key=lambda x: x[1])
        return mejor_resultado


    def fecha_a_excel(self, fecha_str):
        """Convierte una fecha en formato string a número de Excel"""
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y %H:%M")
            base_excel = datetime(1899, 12, 30)
            delta = fecha - base_excel
            return delta.days + delta.seconds / 86400
        except:
            return None

    def es_fecha(self, valor_str):
        """Verifica si un string representa una fecha válida"""
        patron_fecha = r'\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2})?'
        return bool(re.match(patron_fecha, valor_str.strip()))

    def procesar_valor_numerico(self, valor, permitir_negativos=True):
        """Función genérica para procesar valores numéricos con manejo de fechas"""
        if pd.isna(valor) or valor == '' or str(valor).strip() == '':
            return 0.0
        
        valor_str = str(valor).strip()
        
        if '####' in valor_str:
            return 0.0
        
        if self.es_fecha(valor_str):
            if ':' not in valor_str:
                valor_str += ' 0:00'
            fecha_excel = self.fecha_a_excel(valor_str)
            return fecha_excel if fecha_excel is not None else 0.0
        
        valor_str = valor_str.replace('$', '').strip()
        
        if ',' in valor_str and valor_str.count('.') > 0:
            if ',' in valor_str:
                partes = valor_str.split(',')
                parte_entera = partes[0].replace('.', '')
                parte_decimal = partes[1] if len(partes) > 1 else '00'
                valor_str = parte_entera + '.' + parte_decimal
        elif ',' in valor_str and valor_str.count('.') == 0:
            valor_str = valor_str.replace(',', '.')
        
        try:
            resultado = float(valor_str)
            if not permitir_negativos and resultado < 0:
                return 0.0
            return resultado
        except:
            return 0.0

    def procesar_debtor_credit_score(self, valor):
        """Procesa credit score y determina si es una empresa"""
        if pd.isna(valor) or valor == '' or str(valor).strip() == '':
            return 0.0, 0
        
        valor_str = str(valor).strip()
        
        if '####' in valor_str:
            return 0.0, 0
        
        if self.es_fecha(valor_str):
            if ':' not in valor_str:
                valor_str += ' 0:00'
            fecha_excel = self.fecha_a_excel(valor_str)
            if fecha_excel is not None:
                is_company = 1 if fecha_excel < 0 else 0
                credit_score = 0.0 if fecha_excel < 0 else fecha_excel
                return credit_score, is_company
            return 0.0, 0
        
        valor_str = valor_str.replace('$', '').strip()
        valor_str = valor_str.replace(',', '.')
        
        try:
            val_num = float(valor_str)
            is_company = 1 if val_num < 0 else 0
            credit_score = 0.0 if val_num < 0 else val_num
            return credit_score, is_company
        except:
            return 0.0, 0

    def procesar_debtor_level_of_education(self, valor):
        """Procesa nivel de educación"""
        if pd.isna(valor) or valor == '' or str(valor).strip() == '':
            return 'Desconocido'
        
        valor_str = str(valor).strip()
        
        if self.es_fecha(valor_str):
            if ':' not in valor_str:
                valor_str += ' 0:00'
            fecha_excel = self.fecha_a_excel(valor_str)
            if fecha_excel is not None:
                try:
                    codigo = int(fecha_excel) % 10
                    if codigo == 0:
                        codigo = 1
                except:
                    return 'Desconocido'
            else:
                return 'Desconocido'
        else:
            try:
                codigo = int(float(valor_str))
            except:
                return 'Desconocido'
        
        mapeo = {
            1: 'sin estudio',
            2: 'primaria', 
            3: 'bachillerato',
            4: 'técnico profesional',
            5: 'tecnológico',
            6: 'pregrado',
            7: 'especialización',
            8: 'maestría',
            9: 'doctorado'
        }
        
        return mapeo.get(codigo, 'Desconocido')

    def procesar_snr_estimated_unique_properties(self, valor):
        """Procesa propiedades únicas estimadas"""
        if pd.isna(valor) or valor == '' or str(valor).strip() == '':
            return 0.0
        
        valor_str = str(valor).strip()
        
        if valor_str == 'manual_review':
            return 0.0
        
        if self.es_fecha(valor_str):
            if ':' not in valor_str:
                valor_str += ' 0:00'
            fecha_excel = self.fecha_a_excel(valor_str)
            return fecha_excel if fecha_excel is not None else 0.0
        
        try:
            return float(valor_str)
        except:
            return 0.0

    def procesar_calificaciones(self, df):
        """Procesa archivo de calificaciones"""
        df['calificacion_crediticia'] = df['Valor_calificacion'].map(self.mapeo_numeros_a_letras)
        
        for col in self.columnas_excluir:
            if col in df.columns:
                df = df.drop(columns=[col], errors='ignore')
        
        return df

    def transformar_datapoints(self, datapoints_df):
        """Transforma datapoints de formato largo a ancho"""
        # Normalizar esquema (distintas versiones de export) antes de filtrar
        datapoints_df = normalizar_datapoints(datapoints_df)

        # Filtrar los data_name de interés: features del modelo + metadata del titular
        nombres_interes = self.TARGET_DATA_NAMES + self.META_DATA_NAMES
        datapoints_filtrado = datapoints_df[datapoints_df['data_name'].isin(nombres_interes)]

        # Convertir de formato largo a formato ancho
        pivot_df = datapoints_filtrado.pivot_table(
            index=['id_simulacion', 'id_usuario (opcional)', 'id_user_creacion', 'fecha_creacion'],
            columns='data_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        pivot_df.columns.name = None
        
        return pivot_df

    def estandarizar_datos(self, df):
        """Aplica estandarización de datos"""
        # Procesar cada columna según su tipo
        for col in self.TARGET_DATA_NAMES:
            if col in df.columns:
                if col == 'debtor_credit_score':
                    resultados = df[col].apply(self.procesar_debtor_credit_score)
                    df['debtor_credit_score'] = [r[0] for r in resultados]
                    df['is_company'] = [r[1] for r in resultados]
                elif col == 'debtor_level_of_education':
                    df[col] = df[col].apply(self.procesar_debtor_level_of_education)
                    # Variable ORDINAL para el modelo (evita inversiones de importancia
                    # y no rompe si aparece un nivel no visto en entrenamiento).
                    df[EDUCATION_ORDINAL_FEATURE] = df[col].apply(educacion_a_ordinal)
                elif col == 'snr_estimated_unique_properties':
                    df[col] = df[col].apply(self.procesar_snr_estimated_unique_properties)
                else:
                    df[col] = df[col].apply(lambda x: self.procesar_valor_numerico(x, permitir_negativos=False))
        
        # Asegurar que is_company exista
        if 'is_company' not in df.columns:
            df['is_company'] = 0
        
        return df

    def division_segura(self, numerador, denominador):
        """Realiza división segura: si denominador es 0, retorna 0"""
        return np.where(denominador == 0, 0, numerador / denominador)

    def calcular_variables_derivadas(self, df):
        """Calcula variables derivadas"""
        df['equityFactor'] = self.division_segura(
            df['tax_report_total_equity_value'], 
            df['original_amount']
        )
        
        df['quotefactor'] = self.division_segura(
            df['tax_report_total_income_value'] / 12, 
            df['value_monthly_payments']
        )
        
        df['LTVfactor'] = self.division_segura(
            df['original_amount'], 
            df['collateral_commercial_value']
        )
        
        df['bank_inflow_factor'] = self.division_segura(
            df['bank_monthly_avg_credit_value'], 
            df['value_monthly_payments']
        )
        
        df['bank_average_factor'] = self.division_segura(
            df['bank_monthly_avg_blance'], 
            df['value_monthly_payments']
        )
        
        df['creditos_cerrados_factor'] = self.division_segura(
            df['debtor_closed_loan_value'] * 1000, 
            df['original_amount']
        ) - 1
        
        # Reemplazar valores infinitos y nan
        df = df.replace([float('inf'), -float('inf')], 0).fillna(0)
        
        return df

    def filtrar_duplicados(self, df):
        """Filtra duplicados manteniendo el más reciente"""
        # Parseo flexible: soporta dd/mm/yyyy (export viejo) e ISO (export nuevo)
        df['fecha_creacion'] = parsear_fechas(df['fecha_creacion'])
        
        df = df.sort_values('fecha_creacion').drop_duplicates(
            subset=['id_simulacion', 'id_usuario (opcional)'], 
            keep='last'
        )
        
        return df

    def aplicar_filtro_calidad(self, df):
        """Aplica filtro de calidad de datos"""
        condicion_filtro = True
        
        if 'original_amount' in df.columns and 'collateral_commercial_value' in df.columns:
            condicion_filtro = (df['original_amount'] != 0) & (df['collateral_commercial_value'] != 0)
        elif 'original_amount' in df.columns:
            condicion_filtro = (df['original_amount'] != 0)
        elif 'collateral_commercial_value' in df.columns:
            condicion_filtro = (df['collateral_commercial_value'] != 0)
        
        if isinstance(condicion_filtro, bool) and condicion_filtro:
            return df.copy()
        else:
            return df[condicion_filtro].copy()

    def procesar_dataset_completo(self, datapoints_file, calificaciones_file, modo_seleccion='titular'):
        """Procesa un dataset completo (datapoints + calificaciones).
        modo_seleccion: 'titular' (id_type_debtor==1) o 'perfil' (mejor puntaje con pesos)."""
        try:
            print(f"📊 Procesando dataset completo...")
            
            # Lectura ROBUSTA: detecta separador y encoding solo; soporta Excel.
            print(f"📖 Leyendo datapoints...")
            datapoints_df = leer_tabla(
                datapoints_file,
                dtype={'id_simulacion': 'str', 'id_usuario (opcional)': 'str'}
            )

            print(f"📖 Leyendo calificaciones...")
            calificaciones_df = leer_tabla(
                calificaciones_file,
                dtype={'id_simulacion': 'str'}
            )
            
            print(f"📈 Datapoints iniciales: {len(datapoints_df)} registros")
            print(f"📈 Calificaciones iniciales: {len(calificaciones_df)} registros")
            
            # Procesar calificaciones
            calificaciones_df = calificaciones_df.dropna(subset=['Valor_calificacion'])
            calificaciones_df = calificaciones_df.rename(columns={'Idsimulacion': 'id_simulacion'})
            calificaciones_df = self.procesar_calificaciones(calificaciones_df)
            
            print(f"📈 Calificaciones válidas: {len(calificaciones_df)} registros")
            
            # OPTIMIZACIÓN: Filtrar datapoints temprano para reducir memoria
            ids_con_calificacion = set(calificaciones_df['id_simulacion'].dropna().astype(str))
            datapoints_filtrado = datapoints_df[
                datapoints_df['id_simulacion'].astype(str).isin(ids_con_calificacion)
            ]
            
            print(f"📈 Datapoints filtrados: {len(datapoints_filtrado)} registros")
            
            # OPTIMIZACIÓN: Liberar memoria del dataframe original
            del datapoints_df
            
            # Transformar datapoints
            print("🔄 Transformando datapoints...")
            pivot_df = self.transformar_datapoints(datapoints_filtrado)
            
            # OPTIMIZACIÓN: Liberar memoria
            del datapoints_filtrado
            
            print(f"📈 Datos pivoteados: {len(pivot_df)} registros")
            
            # Hacer merge con calificaciones
            print("🔗 Haciendo merge con calificaciones...")
            # Garantizar mismo tipo en la llave (la normalización de columnas
            # ocurre tras la lectura, así que el dtype del read no siempre aplica)
            pivot_df['id_simulacion'] = pivot_df['id_simulacion'].astype(str)
            calificaciones_df['id_simulacion'] = calificaciones_df['id_simulacion'].astype(str)
            datos_df = pivot_df.merge(calificaciones_df, on='id_simulacion', how='left')
            
            # OPTIMIZACIÓN: Liberar memoria
            del pivot_df, calificaciones_df
            
            print(f"📈 Datos después de merge: {len(datos_df)} registros")
            
            # Procesar datos paso a paso para optimizar memoria
            print("🔧 Estandarizando datos...")
            datos_df = self.estandarizar_datos(datos_df)
            
            print("🔍 Aplicando filtro de calidad...")
            datos_df = self.aplicar_filtro_calidad(datos_df)
            
            print("🧮 Calculando variables derivadas...")
            datos_df = self.calcular_variables_derivadas(datos_df)
            
            print("🧹 Filtrando duplicados...")
            datos_df = self.filtrar_duplicados(datos_df)
            
            print(f"📈 Datos finales antes de selección: {len(datos_df)} registros")

            # OPTIMIZACIÓN: Seleccionar deudor principal de manera más eficiente
            print("👑 Seleccionando deudores principales...")
            deudores_seleccionados = []
            
            for id_sim, grupo in datos_df.groupby('id_simulacion'):
                deudor_principal, puntaje_total, puntajes_detalle = self.seleccionar_deudor_principal(grupo, modo=modo_seleccion)

                # Convertir la fila a diccionario si es una Serie
                if hasattr(deudor_principal, 'to_dict'):
                    deudor_dict = deudor_principal.to_dict()
                else:
                    deudor_dict = deudor_principal

                # Agregar el puntaje total
                deudor_dict['calificacion_total'] = round(puntaje_total, 4)
                deudores_seleccionados.append(deudor_dict)

            # Crear DataFrame con deudores seleccionados
            datos_df = pd.DataFrame(deudores_seleccionados)
            
            print(f"✅ Procesamiento completado: {len(datos_df)} deudores principales seleccionados")
            return datos_df
            
        except Exception as e:
            print(f"❌ Error procesando dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_all_data(self, replace_files=False):
        """Procesa todos los datos disponibles con opción de reemplazar archivos permanentes"""
        processed_data = {}
        
        # Verificar archivos disponibles
        datapoints_file = None
        cal_6m_file = None
        cal_12m_file = None
        
        # Si hay archivos temporales y se solicita reemplazo, mover a permanentes
        if replace_files:
            self._replace_permanent_files_with_temp()
        
        # Buscar archivos temporales o existentes (priorizar temporales para procesamiento inmediato)
        if os.path.exists("temp_datapoints.csv"):
            datapoints_file = "temp_datapoints.csv"
        elif os.path.exists("datapoints.csv"):
            datapoints_file = "datapoints.csv"
        
        if os.path.exists("temp_cal_6m.csv"):
            cal_6m_file = "temp_cal_6m.csv"
        elif os.path.exists("Calificaciones6M-119.csv"):
            cal_6m_file = "Calificaciones6M-119.csv"
        
        if os.path.exists("temp_cal_12m.csv"):
            cal_12m_file = "temp_cal_12m.csv"
        elif os.path.exists("Calificaciones12M-110.csv"):
            cal_12m_file = "Calificaciones12M-110.csv"
        
        if not datapoints_file:
            raise Exception("No se encontró archivo de datapoints")
        
        print(f"📁 Usando archivos:")
        print(f"   - Datapoints: {datapoints_file}")
        print(f"   - Calificaciones 6M: {cal_6m_file}")
        print(f"   - Calificaciones 12M: {cal_12m_file}")
        
        # Procesar datos 6M
        if cal_6m_file:
            print("Procesando datos 6 meses...")
            data_6m = self.procesar_dataset_completo(datapoints_file, cal_6m_file)
            if data_6m is not None:
                processed_data['6m'] = data_6m
                print(f"Datos 6M procesados: {len(data_6m)} registros")
                # Guardar deudores seleccionados
                data_6m.to_csv('deudor_seleccionado_6m.csv', sep=';', index=False)
                print(f"Deudores principales 6M guardados: deudor_seleccionado_6m.csv")

        # Procesar datos 12M
        if cal_12m_file:
            print("Procesando datos 12 meses...")
            data_12m = self.procesar_dataset_completo(datapoints_file, cal_12m_file)
            if data_12m is not None:
                processed_data['12m'] = data_12m
                print(f"Datos 12M procesados: {len(data_12m)} registros")
                # Guardar deudores seleccionados
                data_12m.to_csv('deudor_seleccionado_12m.csv', sep=';', index=False)
                print(f"Deudores principales 12M guardados: deudor_seleccionado_12m.csv")
        
        # Limpiar archivos temporales después del procesamiento exitoso
        if replace_files:
            self._cleanup_temp_files()
        
        if not processed_data:
            raise Exception("No se pudieron procesar los datos")
        
        return processed_data
    
    def _replace_permanent_files_with_temp(self):
        """Reemplaza archivos permanentes con archivos temporales si existen"""
        replacements = [
            ("temp_datapoints.csv", "datapoints.csv"),
            ("temp_cal_6m.csv", "Calificaciones6M-119.csv"),
            ("temp_cal_12m.csv", "Calificaciones12M-110.csv")
        ]
        
        for temp_file, permanent_file in replacements:
            if os.path.exists(temp_file):
                try:
                    # Hacer backup del archivo original si existe
                    if os.path.exists(permanent_file):
                        backup_file = f"{permanent_file}.backup"
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                        os.rename(permanent_file, backup_file)
                        print(f"🔄 Backup creado: {backup_file}")
                    
                    # Mover archivo temporal al permanente
                    os.rename(temp_file, permanent_file)
                    print(f"✅ Archivo reemplazado: {permanent_file}")
                    
                except Exception as e:
                    print(f"❌ Error reemplazando {temp_file} -> {permanent_file}: {e}")
    
    def _cleanup_temp_files(self):
        """Limpia archivos temporales después del procesamiento"""
        temp_files = [
            "temp_datapoints.csv",
            "temp_cal_6m.csv",
            "temp_cal_12m.csv"
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Archivo temporal eliminado: {temp_file}")
                except Exception as e:
                    print(f"⚠️ No se pudo eliminar {temp_file}: {e}")

    def cruzar_calificaciones(self, dataset_file, calificaciones_file, output_file):
        """Cruza dataset con calificaciones y guarda resultado"""
        try:
            # Leer dataset principal
            df_dataset = pd.read_csv(dataset_file, sep=';')
            df_dataset.columns = df_dataset.columns.str.strip()
            
            # Leer calificaciones
            df_calificaciones = pd.read_csv(calificaciones_file, sep=',')
            df_calificaciones.columns = df_calificaciones.columns.str.strip()
            df_calificaciones = df_calificaciones.dropna(axis=1, how='all')
            df_calificaciones = df_calificaciones.dropna(subset=['Idsimulacion', 'Valor_calificacion'])
            
            # Realizar merge
            df_dataset['id_simulacion'] = df_dataset['id_simulacion'].astype(str).str.strip()
            df_calificaciones['Idsimulacion'] = df_calificaciones['Idsimulacion'].astype(str).str.strip()
            
            df_resultado = pd.merge(df_dataset, df_calificaciones,
                                  left_on='id_simulacion', right_on='Idsimulacion', how='inner')
            
            # Limpiar y procesar
            if 'Idsimulacion' in df_resultado.columns:
                df_resultado = df_resultado.drop('Idsimulacion', axis=1)
            
            # Actualizar calificación crediticia
            df_resultado['calificacion_crediticia'] = df_resultado['Valor_calificacion'].map(self.mapeo_numeros_a_letras)
            df_resultado['calificacion_crediticia'].fillna('C', inplace=True)
            
            # Guardar resultado
            df_resultado.to_csv(output_file, index=False, sep=';')
            
            print(f"Archivo cruzado guardado: {output_file}")
            print(f"Registros finales: {len(df_resultado)}")
            
            return df_resultado
            
        except Exception as e:
            print(f"Error cruzando calificaciones: {e}")
            return None
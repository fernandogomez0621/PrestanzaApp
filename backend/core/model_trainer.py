import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
# matplotlib/seaborn solo se usan para gráficas (no requeridas por la API).
# Import opcional para no obligar a instalarlas en el contenedor.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None
import warnings
import json
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        
        self.mapeo_numeros_a_letras = {
            10: 'AAA', 9: 'AA', 8: 'A',
            6: 'BBB', 5: 'BB', 4: 'B',
            2: 'CCC', 1: 'CC', 0: 'C'
        }
        
        self.columnas_excluir = [
            'id_simulacion', 'id_usuario (opcional)', 'id_user_creacion', 'IdOperacion',
            'IdOperacion_x', 'IdOperacion_y',
            'fecha_creacion', 'Fecha_calificación (meses)_x', 'Fecha_calificación (meses)_y',
            'Valor_calificacion', 'calificacion_crediticia',
            'Tiene data-points', 'is_company'
        ]
        
        self.scaler_ridge = StandardScaler()
        self.scaler_knn = StandardScaler()
        self.ridge_model = None
        self.knn_model = None
        self.history = None
        self.feature_names = None
        self.input_shape = None
        self.feature_importance = None

    def preprocesar_datos(self, df, es_entrenamiento=True, usar_split=False):
        """Preprocesa los datos para entrenamiento"""
        if 'Valor_calificacion' not in df.columns:
            return None, None, None, None
        
        y = df['Valor_calificacion'].copy()
        X = df.copy()
        
        # Limpiar nombres de columnas
        X.columns = X.columns.str.strip()
        
        # Eliminar columnas Unnamed que pueden aparecer
        unnamed_cols = [col for col in X.columns if 'unnamed' in col.lower()]
        if unnamed_cols:
            print(f"🧹 Eliminando columnas Unnamed: {unnamed_cols}")
            X = X.drop(columns=unnamed_cols)
        
        # Procesar educación como variable ORDINAL (no one-hot).
        # Razones:
        #   - La educación es ordinal; one-hot permitía que el modelo diera más
        #     peso a 'bachillerato' que a 'profesional'. Ordinal lo impide.
        #   - One-hot creaba columnas dummy distintas entre train y predict, y el
        #     modelo fallaba cuando aparecía un nivel no visto (p.ej. 'primaria').
        # data_processor ya genera 'debtor_education_ordinal'. Si no estuviera,
        # lo derivamos del texto aquí mismo.
        from io_utils import educacion_a_ordinal, EDUCATION_ORDINAL_FEATURE
        if EDUCATION_ORDINAL_FEATURE not in X.columns and 'debtor_level_of_education' in X.columns:
            X[EDUCATION_ORDINAL_FEATURE] = X['debtor_level_of_education'].apply(educacion_a_ordinal)
        if 'debtor_level_of_education' in X.columns:
            X = X.drop(columns=['debtor_level_of_education'])
        
        # Excluir columnas no deseadas
        for col in self.columnas_excluir:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # Filtrar columnas problemáticas (incluyendo Unnamed)
        columnas_problematicas = []
        for col in X.columns:
            col_lower = col.lower()
            if (any(patron in col_lower for patron in ['fecha', 'id_', 'meses', 'calificacion', 'unnamed'])
                and not col.startswith('debtor_level_education_')):
                columnas_problematicas.append(col)
        
        if columnas_problematicas:
            print(f"🧹 Eliminando columnas problemáticas: {columnas_problematicas}")
            X = X.drop(columns=columnas_problematicas)
        
        # Convertir columnas object a numéricas
        for col in X.columns:
            if not col.startswith('debtor_level_education_'):
                if X[col].dtype == 'object':
                    try:
                        X[col] = X[col].astype(str).str.replace(',', '.')
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        pass
        
        # Filtrar valores válidos
        mask_valid = ~y.isnull()
        X = X[mask_valid]
        y = y[mask_valid]
        
        # Rellenar valores faltantes
        for col in X.columns:
            if not col.startswith('debtor_level_education_'):
                if X[col].isnull().sum() > 0:
                    if X[col].dtype in ['int64', 'float64']:
                        X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0).astype(int)
        
        # Seleccionar solo columnas numéricas
        X = X.select_dtypes(include=[np.number])
        
        # Guardar información de features
        self.feature_names = list(X.columns)
        self.input_shape = X.shape[1]
        
        if usar_split:
            value_counts = y.value_counts()
            clases_con_una_muestra = value_counts[value_counts == 1]
            
            if len(clases_con_una_muestra) > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            return X_train, X_test, y_train, y_test
        else:
            return X, y

    def entrenar_ridge(self, X_train, y_train):
        """Entrena modelo Ridge con validación cruzada optimizada"""
        print(f"📈 Entrenando Ridge con {len(X_train)} muestras...")
        X_train_scaled = self.scaler_ridge.fit_transform(X_train)
        
        # OPTIMIZACIÓN: Reducir número de alphas para entrenar más rápido
        # pero mantener buena cobertura del espacio de parámetros
        alphas = np.logspace(-4, 2, 100)  # Reducido de 500 a 100
        
        # OPTIMIZACIÓN: Usar menos folds si tenemos pocos datos
        cv_folds = min(5, max(3, len(X_train) // 10))
        print(f"🔄 Usando {cv_folds} folds para validación cruzada")
        
        ridge_cv = RidgeCV(
            alphas=alphas,
            cv=cv_folds,
            scoring='neg_mean_squared_error'
            # Removido store_cv_values ya que no existe en todas las versiones
        )
        ridge_cv.fit(X_train_scaled, y_train)
        
        print(f"✅ Ridge entrenado con alpha óptimo: {ridge_cv.alpha_:.6f}")
        self.ridge_model = ridge_cv
        return ridge_cv

    def entrenar_knn(self, X_train, y_train):
        """Entrena modelo KNN con parámetros optimizados"""
        print(f"🔍 Entrenando KNN con {len(X_train)} muestras...")
        X_train_scaled = self.scaler_knn.fit_transform(X_train)
        
        self.knn_model = KNeighborsRegressor(
            n_neighbors=3,
            weights='distance',
            metric='minkowski'
        )
        
        self.knn_model.fit(X_train_scaled, y_train)
        print(f"✅ KNN entrenado con k=3, weights='distance'")
        return self.knn_model

    def calcular_importancia_caracteristicas(self, X_train, y_train, top_n=15):
        """Calcula importancia de características usando Permutation Importance"""
        if self.knn_model is None:
            print("❌ Modelo KNN no entrenado")
            return None
        
        print(f"🔍 Calculando importancia de características (top {top_n})...")
        X_train_scaled = self.scaler_knn.transform(X_train)
        
        # Calcular permutation importance
        perm_importance = permutation_importance(
            self.knn_model,
            X_train_scaled,
            y_train,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Obtener índices de las características más importantes
        indices = np.argsort(perm_importance.importances_mean)[::-1][:top_n]
        
        # Crear diccionario con resultados
        self.feature_importance = {
            'indices': indices,
            'importances_mean': perm_importance.importances_mean[indices],
            'importances_std': perm_importance.importances_std[indices],
            'feature_names': [self.feature_names[i] for i in indices]
        }
        
        print(f"✅ Importancia calculada para {len(indices)} características")
        return self.feature_importance

    def generar_partial_dependence_plots(self, X_train, y_train, top_n=12):
        """Genera Partial Dependence Plots reales para KNN"""
        if self.knn_model is None or self.feature_importance is None:
            print("❌ Modelo KNN o importancia de características no disponibles")
            return None
        
        print(f"📊 Generando Partial Dependence Plots para top {top_n} características...")
        
        # Obtener las características más importantes
        top_features_indices = self.feature_importance['indices'][:top_n]
        top_features_names = [self.feature_names[i] for i in top_features_indices]
        
        # Escalar datos de entrenamiento
        X_train_scaled = self.scaler_knn.transform(X_train)
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        
        # Generar PDP data para cada característica
        pdp_data = {}
        
        for i, feature_idx in enumerate(top_features_indices):
            feature_name = self.feature_names[feature_idx]
            
            # Crear rango de valores para la característica
            feature_values = X_train_df.iloc[:, feature_idx]
            min_val, max_val = feature_values.min(), feature_values.max()
            
            # Crear grid de valores
            grid_values = np.linspace(min_val, max_val, 50)
            
            # Calcular dependencia parcial
            pdp_values = []
            
            for grid_val in grid_values:
                # Crear copia de los datos
                X_modified = X_train_df.copy()
                # Cambiar solo la característica de interés
                X_modified.iloc[:, feature_idx] = grid_val
                
                # Predecir con el modelo
                predictions = self.knn_model.predict(X_modified.values)
                # Promedio de las predicciones
                pdp_values.append(np.mean(predictions))
            
            # Guardar datos del PDP
            pdp_data[feature_name] = {
                'grid_values': grid_values.tolist(),
                'pdp_values': pdp_values,
                'feature_mean': float(X_train[feature_name].mean()),
                'feature_std': float(X_train[feature_name].std()),
                'importance': float(self.feature_importance['importances_mean'][i])
            }
        
        print(f"✅ PDP generado para {len(pdp_data)} características")
        return pdp_data

    def mostrar_escalado_info(self, X_train, top_n=15):
        """Muestra información sobre el escalado de las características"""
        if self.feature_importance is None:
            return None
        
        top_features_indices = self.feature_importance['indices'][:top_n]
        
        escalado_info = {}
        
        for i, feature_idx in enumerate(top_features_indices):
            feature_name = self.feature_names[feature_idx]
            
            escalado_info[feature_name] = {
                'mean': float(X_train[feature_name].mean()),
                'std': float(X_train[feature_name].std()),
                'min': float(X_train[feature_name].min()),
                'max': float(X_train[feature_name].max()),
                'importance_rank': i + 1,
                'importance_value': float(self.feature_importance['importances_mean'][i])
            }
        
        return escalado_info

    def generar_matrices_confusion_train_test(self, X_train, y_train, test_size=0.2):
        """Genera matrices de confusión para entrenamiento y test del modelo KNN"""
        if self.knn_model is None:
            print("❌ Modelo KNN no entrenado")
            return None
        
        print(f"📊 Generando matrices de confusión train/test para KNN...")
        
        # Dividir datos en train/test
        try:
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
            )
        except:
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=test_size, random_state=42
            )
        
        # Escalar datos
        X_train_scaled = self.scaler_knn.fit_transform(X_train_split)
        X_test_scaled = self.scaler_knn.transform(X_test_split)
        
        # Entrenar modelo temporal para train/test
        knn_temp = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='minkowski')
        knn_temp.fit(X_train_scaled, y_train_split)
        
        # Predicciones
        y_train_pred = knn_temp.predict(X_train_scaled)
        y_test_pred = knn_temp.predict(X_test_scaled)
        
        # Redondear a valores válidos
        valores_validos = list(self.mapeo_numeros_a_letras.keys())
        def redondear_a_valores_validos(pred):
            return np.array([min(valores_validos, key=lambda x: abs(x - p)) for p in pred])
        
        y_train_pred_round = redondear_a_valores_validos(y_train_pred)
        y_test_pred_round = redondear_a_valores_validos(y_test_pred)
        
        # Convertir a etiquetas
        def convertir_a_etiquetas(valores):
            return np.array([self.mapeo_numeros_a_letras.get(v, 'C') for v in valores])
        
        y_train_labels = convertir_a_etiquetas(y_train_split)
        y_test_labels = convertir_a_etiquetas(y_test_split)
        pred_train_labels = convertir_a_etiquetas(y_train_pred_round)
        pred_test_labels = convertir_a_etiquetas(y_test_pred_round)
        
        # Obtener etiquetas presentes
        etiquetas_presentes = sorted(set(np.concatenate([y_train_labels, y_test_labels])))
        
        # Crear matrices de confusión
        cm_train = confusion_matrix(y_train_labels, pred_train_labels, labels=etiquetas_presentes)
        cm_test = confusion_matrix(y_test_labels, pred_test_labels, labels=etiquetas_presentes)
        
        # Calcular métricas
        train_accuracy = np.mean(y_train_split == y_train_pred_round)
        test_accuracy = np.mean(y_test_split == y_test_pred_round)
        
        matrices_data = {
            'train_matrix': cm_train,
            'test_matrix': cm_test,
            'labels': etiquetas_presentes,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_samples': len(y_train_split),
            'test_samples': len(y_test_split)
        }
        
        print(f"✅ Matrices generadas - Train: {cm_train.shape}, Test: {cm_test.shape}")
        print(f"📊 Accuracy - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
        
        return matrices_data

    def evaluar_modelos(self, X_train, y_train, incluir_knn=False):
        """Evalúa modelos disponibles y retorna métricas"""
        valores_validos = list(self.mapeo_numeros_a_letras.keys())
        
        def redondear_a_valores_validos(pred):
            return np.array([min(valores_validos, key=lambda x: abs(x - p)) for p in pred])
        
        resultados = {}
        predicciones = {}
        
        # Ridge (si existe)
        if self.ridge_model is not None:
            X_train_scaled_ridge = self.scaler_ridge.transform(X_train)
            y_train_pred_ridge = self.ridge_model.predict(X_train_scaled_ridge)
            y_train_pred_ridge_round = redondear_a_valores_validos(y_train_pred_ridge)
            
            ridge_results = {
                'train_mse': mean_squared_error(y_train, y_train_pred_ridge),
                'train_r2': r2_score(y_train, y_train_pred_ridge),
                'train_accuracy': np.mean(y_train == y_train_pred_ridge_round)
            }
            
            resultados['ridge'] = ridge_results
            predicciones['ridge'] = {
                'train': y_train_pred_ridge,
                'train_round': y_train_pred_ridge_round
            }
        
        # KNN (si existe y se solicita)
        if incluir_knn and self.knn_model is not None:
            X_train_scaled_knn = self.scaler_knn.transform(X_train)
            y_train_pred_knn = self.knn_model.predict(X_train_scaled_knn)
            y_train_pred_knn_round = redondear_a_valores_validos(y_train_pred_knn)
            
            knn_results = {
                'train_mse': mean_squared_error(y_train, y_train_pred_knn),
                'train_r2': r2_score(y_train, y_train_pred_knn),
                'train_accuracy': np.mean(y_train == y_train_pred_knn_round)
            }
            
            resultados['knn'] = knn_results
            predicciones['knn'] = {
                'train': y_train_pred_knn,
                'train_round': y_train_pred_knn_round
            }
        
        return resultados, predicciones

    def generar_matriz_confusion(self, y_train, predicciones):
        """Genera matrices de confusión SOLO con datos reales presentes"""
        def convertir_a_etiquetas(valores):
            return np.array([self.mapeo_numeros_a_letras.get(v, 'C') for v in valores])
        
        y_train_labels = convertir_a_etiquetas(y_train)
        
        # OPTIMIZACIÓN: Solo usar etiquetas que realmente están presentes en los datos
        etiquetas_reales = sorted(set(y_train_labels))
        print(f"🎯 Etiquetas reales en datos de entrenamiento: {etiquetas_reales}")
        
        matrices = {}
        
        for modelo in ['ridge', 'knn']:
            if modelo in predicciones:
                preds = predicciones[modelo]
                pred_train_labels = convertir_a_etiquetas(preds['train_round'])
                
                # OPTIMIZACIÓN: Solo incluir etiquetas que están en datos reales
                etiquetas_predichas = set(pred_train_labels)
                etiquetas_finales = sorted(set(etiquetas_reales) | etiquetas_predichas)
                
                cm_train = confusion_matrix(y_train_labels, pred_train_labels, labels=etiquetas_finales)
                
                # VALIDACIÓN: Verificar que la matriz tiene el tamaño correcto
                total_muestras = len(y_train_labels)
                suma_matriz = cm_train.sum()
                
                print(f"📊 Matriz {modelo}: {cm_train.shape}, muestras totales: {total_muestras}, suma matriz: {suma_matriz}")
                
                if suma_matriz != total_muestras:
                    print(f"⚠️ ADVERTENCIA: Inconsistencia en matriz {modelo} - suma no coincide con total de muestras")
                
                matrices[modelo] = {
                    'matrix': cm_train,
                    'labels': etiquetas_finales,
                    'total_samples': total_muestras,
                    'matrix_sum': suma_matriz
                }
        
        return matrices

    def plotear_curvas_entrenamiento(self):
        """Genera gráficos de curvas de entrenamiento"""
        if self.history is None:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(self.history.history['loss']) + 1)
        
        ax1.plot(epochs, self.history.history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Neural Network Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history.history['mae'], 'b-', label='Training MAE')
        ax2.plot(epochs, self.history.history['val_mae'], 'r-', label='Validation MAE')
        ax2.set_title('Neural Network MAE')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def exportar_modelos(self, tipo_modelo, output_dir, resultados, predicciones=None, y_train=None):
        """Exporta modelos completos con metadatos"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metadatos
        metadata = {
            "timestamp": timestamp,
            "tipo_modelo": tipo_modelo,
            "feature_names": self.feature_names,
            "input_shape": self.input_shape,
            "mapeo_numeros_a_letras": self.mapeo_numeros_a_letras,
            "columnas_excluidas": self.columnas_excluir,
            "random_seed": 42
        }
        
        # Métricas
        metricas_export = {
            "resultados": resultados,
            "timestamp": timestamp,
            "tipo_evaluacion": "full_training"
        }
        
        # Configuración
        configuracion = {
            "ridge": {
                "alphas_range": [-4, 2],
                "n_alphas": 500,
                "cv_folds": 5,
                "scoring": "neg_mean_squared_error",
                "alpha_seleccionado": float(self.ridge_model.alpha_) if self.ridge_model else None
            }
        }
        
        # Historial de entrenamiento
        historial_entrenamiento = None
        if self.history:
            historial_entrenamiento = {
                "history": {k: [float(v) for v in vals] for k, vals in self.history.history.items()},
                "epochs_completed": len(self.history.history['loss']),
                "best_epoch": int(np.argmin(self.history.history['val_loss'])) + 1 if 'val_loss' in self.history.history else None
            }
        
        # Guardar archivos JSON
        with open(f"{output_dir}/metadata_{tipo_modelo}.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        with open(f"{output_dir}/metricas_{tipo_modelo}.json", 'w', encoding='utf-8') as f:
            json.dump(metricas_export, f, indent=2, ensure_ascii=False)
        
        with open(f"{output_dir}/configuracion_{tipo_modelo}.json", 'w', encoding='utf-8') as f:
            json.dump(configuracion, f, indent=2, ensure_ascii=False)
        
        if historial_entrenamiento:
            with open(f"{output_dir}/historial_entrenamiento_{tipo_modelo}.json", 'w', encoding='utf-8') as f:
                json.dump(historial_entrenamiento, f, indent=2, ensure_ascii=False)
        
        # Guardar modelos y scalers
        if self.ridge_model:
            with open(f"{output_dir}/ridge_model_{tipo_modelo}.pkl", 'wb') as f:
                pickle.dump(self.ridge_model, f)
            
            with open(f"{output_dir}/scaler_ridge_{tipo_modelo}.pkl", 'wb') as f:
                pickle.dump(self.scaler_ridge, f)
        
        if self.knn_model:
            with open(f"{output_dir}/knn_model_{tipo_modelo}.pkl", 'wb') as f:
                pickle.dump(self.knn_model, f)
            
            with open(f"{output_dir}/scaler_knn_{tipo_modelo}.pkl", 'wb') as f:
                pickle.dump(self.scaler_knn, f)
        
        # Guardar importancia de características si existe
        if self.feature_importance:
            with open(f"{output_dir}/feature_importance_{tipo_modelo}.json", 'w', encoding='utf-8') as f:
                # Convertir numpy arrays a listas para JSON
                importance_data = {
                    'indices': self.feature_importance['indices'].tolist(),
                    'importances_mean': self.feature_importance['importances_mean'].tolist(),
                    'importances_std': self.feature_importance['importances_std'].tolist(),
                    'feature_names': self.feature_importance['feature_names']
                }
                json.dump(importance_data, f, indent=2, ensure_ascii=False)
        
        return output_dir

    def train_models_6m(self, data_6m, epochs=100):
        """Entrena modelos para 6 meses: Ridge + KNN"""
        X_train, y_train = self.preprocesar_datos(data_6m, usar_split=False)

        if X_train is None or y_train is None:
            raise Exception("Error en preprocesamiento de datos 6M")

        print("🔥 Entrenando 2 modelos para 6 meses: Ridge + KNN")

        # Entrenar modelos
        self.entrenar_ridge(X_train, y_train)
        self.entrenar_knn(X_train, y_train)

        # Evaluar (incluyendo KNN)
        resultados, predicciones = self.evaluar_modelos(X_train, y_train, incluir_knn=True)

        # Calcular importancia de características para KNN
        feature_importance = self.calcular_importancia_caracteristicas(X_train, y_train, top_n=15)

        # Generar Partial Dependence Plots para KNN
        pdp_data = self.generar_partial_dependence_plots(X_train, y_train, top_n=12)
        
        # Generar información de escalado
        escalado_info = self.mostrar_escalado_info(X_train, top_n=15)

        # Generar matrices de confusión reales (entrenamiento completo)
        matrices_confusion = self.generar_matriz_confusion(y_train, predicciones)
        
        # Generar matrices de confusión train/test para KNN
        matrices_train_test = self.generar_matrices_confusion_train_test(X_train, y_train, test_size=0.2)

        # Exportar
        output_dir = self.exportar_modelos("6M", "modelos_exportados/6_meses", resultados, predicciones, y_train)

        # Crear objeto de modelos para retornar
        models = {
            'ridge_model': self.ridge_model,
            'knn_model': self.knn_model,
            'scaler_ridge': self.scaler_ridge,
            'scaler_knn': self.scaler_knn,
            'feature_names': self.feature_names,
            'feature_importance': feature_importance,
            'pdp_data': pdp_data,
            'escalado_info': escalado_info,
            'matrices_train_test': matrices_train_test,
            'metadata': {
                'input_shape': self.input_shape,
                'mapeo_numeros_a_letras': self.mapeo_numeros_a_letras
            },
            'matrices_confusion': matrices_confusion
        }

        return models, resultados

    def train_models_12m(self, data_12m, epochs=300):
        """Entrena SOLO modelo KNN para 12 meses"""
        X_train, y_train = self.preprocesar_datos(data_12m, usar_split=False)

        if X_train is None or y_train is None:
            raise Exception("Error en preprocesamiento de datos 12M")

        print("🔥 Entrenando SOLO modelo KNN para 12 meses")

        # Limpiar modelos anteriores para 12M
        self.ridge_model = None
        
        # Entrenar solo KNN
        self.entrenar_knn(X_train, y_train)

        # Evaluar (solo KNN)
        resultados, predicciones = self.evaluar_modelos(X_train, y_train, incluir_knn=True)

        # Calcular importancia de características para KNN
        feature_importance = self.calcular_importancia_caracteristicas(X_train, y_train, top_n=15)

        # Generar Partial Dependence Plots para KNN
        pdp_data = self.generar_partial_dependence_plots(X_train, y_train, top_n=12)
        
        # Generar información de escalado
        escalado_info = self.mostrar_escalado_info(X_train, top_n=15)

        # Generar matrices de confusión reales
        matrices_confusion = self.generar_matriz_confusion(y_train, predicciones)
        
        # Generar matrices de confusión train/test para KNN
        matrices_train_test = self.generar_matrices_confusion_train_test(X_train, y_train, test_size=0.2)

        # Exportar
        output_dir = self.exportar_modelos("12M", "modelos_exportados/12_meses", resultados, predicciones, y_train)

        # Crear objeto de modelos para retornar (solo KNN)
        models = {
            'knn_model': self.knn_model,
            'scaler_knn': self.scaler_knn,
            'feature_names': self.feature_names,
            'feature_importance': feature_importance,
            'pdp_data': pdp_data,
            'escalado_info': escalado_info,
            'matrices_train_test': matrices_train_test,
            'metadata': {
                'input_shape': self.input_shape,
                'mapeo_numeros_a_letras': self.mapeo_numeros_a_letras,
                'modelo_tipo': 'KNN_only'
            },
            'matrices_confusion': matrices_confusion
        }

        return models, resultados
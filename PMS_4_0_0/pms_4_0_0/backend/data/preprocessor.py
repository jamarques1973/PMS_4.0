"""
Preprocesador de Datos - PMS 4.0.0
=================================

Este módulo se encarga del preprocesamiento de datos, incluyendo limpieza,
transformación, escalado, codificación y preparación para modelado.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold

from ...controller.logger import Logger
from ...controller.config import Config


class DataPreprocessor:
    """
    Preprocesador de datos para PMS 4.0.0
    
    Responsabilidades:
    - Limpieza de datos
    - Manejo de valores faltantes
    - Escalado y normalización
    - Codificación de variables categóricas
    - Transformación de características
    """
    
    def __init__(self, config: Config, logger: Logger):
        """
        Inicializa el preprocesador de datos
        
        Args:
            config: Configuración del sistema
            logger: Sistema de logging
        """
        self.config = config
        self.logger = logger
        
        # Configuración de preprocesamiento
        self.preprocessing_config = {
            'missing_strategy': 'auto',  # 'auto', 'drop', 'impute'
            'imputation_method': 'mean',  # 'mean', 'median', 'mode', 'knn'
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
            'encoding_method': 'auto',  # 'auto', 'label', 'onehot', 'none'
            'remove_duplicates': True,
            'remove_outliers': False,
            'outlier_method': 'iqr',  # 'iqr', 'zscore'
            'feature_selection': False,
            'variance_threshold': 0.01
        }
        
        # Actualizar con configuración del sistema
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'preprocessing'):
            self.preprocessing_config.update(self.config.data.preprocessing.__dict__)
        
        # Fitted transformers
        self.fitted_transformers = {}
        
        self.logger.info("Preprocesador de datos inicializado correctamente")
    
    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocesa un dataset completo
        
        Args:
            data: DataFrame a preprocesar
            **kwargs: Parámetros adicionales de preprocesamiento
            
        Returns:
            DataFrame preprocesado
        """
        with self.logger.operation_trace("preprocesamiento_completo"):
            self.logger.info(f"Preprocesando dataset: {data.shape}")
            
            # Combinar configuración con parámetros adicionales
            config = {**self.preprocessing_config, **kwargs}
            
            try:
                # Crear copia para no modificar el original
                processed_data = data.copy()
                
                # 1. Limpieza básica
                processed_data = self._clean_data(processed_data, config)
                
                # 2. Manejo de duplicados
                if config.get('remove_duplicates', True):
                    processed_data = self._remove_duplicates(processed_data)
                
                # 3. Manejo de valores faltantes
                processed_data = self._handle_missing_values(processed_data, config)
                
                # 4. Manejo de outliers
                if config.get('remove_outliers', False):
                    processed_data = self._handle_outliers(processed_data, config)
                
                # 5. Codificación de variables categóricas
                processed_data = self._encode_categorical_variables(processed_data, config)
                
                # 6. Escalado de variables numéricas
                processed_data = self._scale_numeric_variables(processed_data, config)
                
                # 7. Selección de características
                if config.get('feature_selection', False):
                    processed_data = self._select_features(processed_data, config)
                
                self.logger.info(f"Preprocesamiento completado: {processed_data.shape}")
                
                return processed_data
                
            except Exception as e:
                self.logger.log_exception(e, "preprocesamiento_completo")
                raise
    
    def _clean_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Limpia los datos básicos"""
        cleaned_data = data.copy()
        
        # Limpiar nombres de columnas
        cleaned_data.columns = [str(col).strip().replace(' ', '_').lower() for col in cleaned_data.columns]
        
        # Eliminar columnas completamente vacías
        empty_cols = cleaned_data.columns[cleaned_data.isnull().all()].tolist()
        if empty_cols:
            cleaned_data = cleaned_data.drop(columns=empty_cols)
            self.logger.info(f"Eliminadas columnas vacías: {empty_cols}")
        
        # Eliminar filas completamente vacías
        empty_rows = cleaned_data.isnull().all(axis=1)
        if empty_rows.any():
            cleaned_data = cleaned_data.dropna(how='all')
            self.logger.info(f"Eliminadas {empty_rows.sum()} filas completamente vacías")
        
        return cleaned_data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas duplicadas"""
        initial_rows = len(data)
        cleaned_data = data.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_data)
        
        if removed_rows > 0:
            self.logger.info(f"Eliminadas {removed_rows} filas duplicadas")
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Maneja valores faltantes"""
        strategy = config.get('missing_strategy', 'auto')
        method = config.get('imputation_method', 'mean')
        
        if strategy == 'drop':
            # Eliminar filas con valores faltantes
            initial_rows = len(data)
            cleaned_data = data.dropna()
            removed_rows = initial_rows - len(cleaned_data)
            
            if removed_rows > 0:
                self.logger.info(f"Eliminadas {removed_rows} filas con valores faltantes")
            
            return cleaned_data
        
        elif strategy == 'impute' or strategy == 'auto':
            # Imputar valores faltantes
            cleaned_data = data.copy()
            
            # Separar columnas numéricas y categóricas
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            categorical_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns
            
            # Imputar columnas numéricas
            if len(numeric_cols) > 0:
                if method == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    cleaned_data[numeric_cols] = imputer.fit_transform(cleaned_data[numeric_cols])
                    self.fitted_transformers['numeric_imputer'] = imputer
                else:
                    imputer = SimpleImputer(strategy=method)
                    cleaned_data[numeric_cols] = imputer.fit_transform(cleaned_data[numeric_cols])
                    self.fitted_transformers['numeric_imputer'] = imputer
                
                self.logger.info(f"Imputados valores faltantes en {len(numeric_cols)} columnas numéricas usando {method}")
            
            # Imputar columnas categóricas
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                cleaned_data[categorical_cols] = imputer.fit_transform(cleaned_data[categorical_cols])
                self.fitted_transformers['categorical_imputer'] = imputer
                
                self.logger.info(f"Imputados valores faltantes en {len(categorical_cols)} columnas categóricas")
            
            return cleaned_data
        
        else:
            return data
    
    def _handle_outliers(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Maneja outliers"""
        method = config.get('outlier_method', 'iqr')
        cleaned_data = data.copy()
        
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                outliers = z_scores > 3
            
            if outliers.any():
                # Reemplazar outliers con los límites
                cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                
                self.logger.info(f"Tratados {outliers.sum()} outliers en columna '{col}' usando {method}")
        
        return cleaned_data
    
    def _encode_categorical_variables(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Codifica variables categóricas"""
        method = config.get('encoding_method', 'auto')
        encoded_data = data.copy()
        
        categorical_cols = encoded_data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return encoded_data
        
        for col in categorical_cols:
            unique_count = encoded_data[col].nunique()
            
            if method == 'auto':
                # Decidir automáticamente el método
                if unique_count <= 10:
                    encoding_method = 'onehot'
                else:
                    encoding_method = 'label'
            else:
                encoding_method = method
            
            if encoding_method == 'label':
                # Label encoding
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                self.fitted_transformers[f'label_encoder_{col}'] = le
                
                self.logger.info(f"Label encoding aplicado a '{col}' ({unique_count} categorías)")
                
            elif encoding_method == 'onehot':
                # One-hot encoding
                if unique_count <= 10:  # Solo si hay pocas categorías
                    dummies = pd.get_dummies(encoded_data[col], prefix=col)
                    encoded_data = pd.concat([encoded_data, dummies], axis=1)
                    encoded_data = encoded_data.drop(columns=[col])
                    
                    self.logger.info(f"One-hot encoding aplicado a '{col}' ({unique_count} categorías)")
                else:
                    # Si hay muchas categorías, usar label encoding
                    le = LabelEncoder()
                    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                    self.fitted_transformers[f'label_encoder_{col}'] = le
                    
                    self.logger.info(f"Label encoding aplicado a '{col}' (muchas categorías: {unique_count})")
        
        return encoded_data
    
    def _scale_numeric_variables(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Escala variables numéricas"""
        method = config.get('scaling_method', 'standard')
        scaled_data = data.copy()
        
        numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0 or method == 'none':
            return scaled_data
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return scaled_data
        
        scaled_data[numeric_cols] = scaler.fit_transform(scaled_data[numeric_cols])
        self.fitted_transformers['scaler'] = scaler
        
        self.logger.info(f"Escalado aplicado a {len(numeric_cols)} columnas numéricas usando {method}")
        
        return scaled_data
    
    def _select_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Selecciona características basadas en varianza"""
        threshold = config.get('variance_threshold', 0.01)
        
        # Seleccionar solo columnas numéricas para selección de características
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
        
        selector = VarianceThreshold(threshold=threshold)
        selected_features = selector.fit_transform(data[numeric_cols])
        
        # Obtener nombres de características seleccionadas
        selected_cols = numeric_cols[selector.get_support()]
        
        # Crear nuevo DataFrame con características seleccionadas
        selected_data = pd.DataFrame(selected_features, columns=selected_cols, index=data.index)
        
        # Agregar columnas no numéricas de vuelta
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        self.fitted_transformers['feature_selector'] = selector
        
        removed_features = len(numeric_cols) - len(selected_cols)
        if removed_features > 0:
            self.logger.info(f"Eliminadas {removed_features} características con baja varianza (threshold: {threshold})")
        
        return selected_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones previamente ajustadas a nuevos datos
        
        Args:
            data: DataFrame a transformar
            
        Returns:
            DataFrame transformado
        """
        if not self.fitted_transformers:
            raise ValueError("No hay transformadores ajustados. Ejecute preprocess() primero.")
        
        transformed_data = data.copy()
        
        # Aplicar imputación
        if 'numeric_imputer' in self.fitted_transformers:
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                transformed_data[numeric_cols] = self.fitted_transformers['numeric_imputer'].transform(transformed_data[numeric_cols])
        
        if 'categorical_imputer' in self.fitted_transformers:
            categorical_cols = transformed_data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                transformed_data[categorical_cols] = self.fitted_transformers['categorical_imputer'].transform(transformed_data[categorical_cols])
        
        # Aplicar codificación
        for key, transformer in self.fitted_transformers.items():
            if key.startswith('label_encoder_'):
                col = key.replace('label_encoder_', '')
                if col in transformed_data.columns:
                    transformed_data[col] = transformer.transform(transformed_data[col].astype(str))
        
        # Aplicar escalado
        if 'scaler' in self.fitted_transformers:
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                transformed_data[numeric_cols] = self.fitted_transformers['scaler'].transform(transformed_data[numeric_cols])
        
        # Aplicar selección de características
        if 'feature_selector' in self.fitted_transformers:
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_features = self.fitted_transformers['feature_selector'].transform(transformed_data[numeric_cols])
                selected_cols = numeric_cols[self.fitted_transformers['feature_selector'].get_support()]
                
                # Crear nuevo DataFrame
                transformed_data = pd.DataFrame(selected_features, columns=selected_cols, index=transformed_data.index)
                
                # Agregar columnas no numéricas de vuelta
                non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_cols) > 0:
                    transformed_data = pd.concat([transformed_data, data[non_numeric_cols]], axis=1)
        
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Ajusta y transforma los datos en una sola operación
        
        Args:
            data: DataFrame a procesar
            **kwargs: Parámetros de preprocesamiento
            
        Returns:
            DataFrame procesado
        """
        return self.preprocess(data, **kwargs)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Revierte las transformaciones aplicadas
        
        Args:
            data: DataFrame transformado
            
        Returns:
            DataFrame con transformaciones revertidas
        """
        if not self.fitted_transformers:
            raise ValueError("No hay transformadores ajustados.")
        
        original_data = data.copy()
        
        # Revertir escalado
        if 'scaler' in self.fitted_transformers:
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                original_data[numeric_cols] = self.fitted_transformers['scaler'].inverse_transform(original_data[numeric_cols])
        
        # Revertir codificación
        for key, transformer in self.fitted_transformers.items():
            if key.startswith('label_encoder_'):
                col = key.replace('label_encoder_', '')
                if col in original_data.columns:
                    original_data[col] = transformer.inverse_transform(original_data[col])
        
        return original_data
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del preprocesamiento aplicado
        
        Returns:
            Resumen del preprocesamiento
        """
        summary = {
            'fitted_transformers': list(self.fitted_transformers.keys()),
            'preprocessing_config': self.preprocessing_config,
            'transformer_info': {}
        }
        
        for key, transformer in self.fitted_transformers.items():
            summary['transformer_info'][key] = {
                'type': type(transformer).__name__,
                'parameters': getattr(transformer, 'get_params', lambda: {})()
            }
        
        return summary
    
    def save_transformers(self, file_path: str):
        """
        Guarda los transformadores ajustados
        
        Args:
            file_path: Ruta donde guardar los transformadores
        """
        import joblib
        
        try:
            joblib.dump(self.fitted_transformers, file_path)
            self.logger.info(f"Transformadores guardados en: {file_path}")
        except Exception as e:
            self.logger.log_exception(e, "guardar_transformadores")
            raise
    
    def load_transformers(self, file_path: str):
        """
        Carga transformadores previamente guardados
        
        Args:
            file_path: Ruta de los transformadores
        """
        import joblib
        
        try:
            self.fitted_transformers = joblib.load(file_path)
            self.logger.info(f"Transformadores cargados desde: {file_path}")
        except Exception as e:
            self.logger.log_exception(e, "cargar_transformadores")
            raise
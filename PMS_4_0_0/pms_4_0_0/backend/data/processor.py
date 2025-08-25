"""
Procesador Principal de Datos - PMS 4.0.0
========================================

Este módulo actúa como el coordinador principal para todas las operaciones
relacionadas con datos, incluyendo carga, validación, preprocesamiento
y feature engineering.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import joblib
from datetime import datetime

from .loader import DataLoader
from .validator import DataValidator
from .preprocessor import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .exploratory import ExploratoryAnalyzer
from ...controller.logger import Logger
from ...controller.config import Config


class DataProcessor:
    """
    Procesador principal de datos para PMS 4.0.0
    
    Responsabilidades:
    - Coordinación de todas las operaciones de datos
    - Gestión del flujo de datos a través del pipeline
    - Caché y optimización de operaciones
    - Validación y control de calidad
    """
    
    def __init__(self, config: Config, logger: Logger):
        """
        Inicializa el procesador de datos
        
        Args:
            config: Configuración del sistema
            logger: Sistema de logging
        """
        self.config = config
        self.logger = logger
        
        # Inicializar componentes
        self.loader = DataLoader(config, logger)
        self.validator = DataValidator(config, logger)
        self.preprocessor = DataPreprocessor(config, logger)
        self.feature_engineer = FeatureEngineer(config, logger)
        self.exploratory_analyzer = ExploratoryAnalyzer(config, logger)
        
        # Estado del procesador
        self.data_cache = {}
        self.processing_history = []
        self.current_dataset = None
        
        # Crear directorios necesarios
        self._create_directories()
        
        self.logger.info("Procesador de datos inicializado correctamente")
    
    def _create_directories(self):
        """Crea los directorios necesarios para el procesamiento de datos"""
        dirs = [
            self.config.data.cache_dir,
            self.config.data.temp_dir,
            self.config.data.output_dir,
            self.config.data.backup_dir
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Carga datos desde un archivo
        
        Args:
            file_path: Ruta al archivo de datos
            **kwargs: Parámetros adicionales para la carga
            
        Returns:
            Diccionario con los datos cargados y metadatos
        """
        with self.logger.operation_trace("carga_datos"):
            self.logger.info(f"Cargando datos desde: {file_path}")
            
            try:
                # Cargar datos
                data_info = self.loader.load_file(file_path, **kwargs)
                
                # Validar datos cargados
                validation_result = self.validator.validate_dataset(data_info['data'])
                
                if not validation_result['is_valid']:
                    self.logger.warning(f"Problemas de validación detectados: {validation_result['issues']}")
                
                # Guardar en caché
                cache_key = f"loaded_{Path(file_path).stem}"
                self.data_cache[cache_key] = {
                    'data': data_info['data'],
                    'metadata': data_info['metadata'],
                    'validation': validation_result,
                    'loaded_at': datetime.now().isoformat()
                }
                
                self.current_dataset = cache_key
                
                # Registrar en historial
                self.processing_history.append({
                    'operation': 'load_data',
                    'file_path': file_path,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'data_shape': data_info['data'].shape,
                    'validation_issues': len(validation_result.get('issues', []))
                })
                
                self.logger.info(f"Datos cargados exitosamente: {data_info['data'].shape}")
                
                return {
                    'data': data_info['data'],
                    'metadata': data_info['metadata'],
                    'validation': validation_result,
                    'cache_key': cache_key
                }
                
            except Exception as e:
                self.logger.log_exception(e, "carga_datos")
                raise
    
    def preprocess_data(self, data_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Preprocesa los datos cargados
        
        Args:
            data_key: Clave de los datos en caché (si None, usa el dataset actual)
            **kwargs: Parámetros de preprocesamiento
            
        Returns:
            Diccionario con los datos preprocesados
        """
        with self.logger.operation_trace("preprocesamiento_datos"):
            if data_key is None:
                data_key = self.current_dataset
            
            if data_key not in self.data_cache:
                raise ValueError(f"Datos no encontrados en caché: {data_key}")
            
            self.logger.info(f"Preprocesando datos: {data_key}")
            
            try:
                original_data = self.data_cache[data_key]['data']
                
                # Aplicar preprocesamiento
                preprocessed_data = self.preprocessor.preprocess(
                    original_data, 
                    **kwargs
                )
                
                # Validar datos preprocesados
                validation_result = self.validator.validate_preprocessed_data(preprocessed_data)
                
                # Guardar en caché
                cache_key = f"preprocessed_{data_key}"
                self.data_cache[cache_key] = {
                    'data': preprocessed_data,
                    'metadata': {
                        'original_key': data_key,
                        'preprocessing_params': kwargs,
                        'preprocessed_at': datetime.now().isoformat()
                    },
                    'validation': validation_result
                }
                
                # Registrar en historial
                self.processing_history.append({
                    'operation': 'preprocess_data',
                    'original_key': data_key,
                    'preprocessed_key': cache_key,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'original_shape': original_data.shape,
                    'preprocessed_shape': preprocessed_data.shape,
                    'validation_issues': len(validation_result.get('issues', []))
                })
                
                self.logger.info(f"Datos preprocesados exitosamente: {preprocessed_data.shape}")
                
                return {
                    'data': preprocessed_data,
                    'metadata': self.data_cache[cache_key]['metadata'],
                    'validation': validation_result,
                    'cache_key': cache_key
                }
                
            except Exception as e:
                self.logger.log_exception(e, "preprocesamiento_datos")
                raise
    
    def select_features(self, data_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Realiza selección de características
        
        Args:
            data_key: Clave de los datos en caché
            **kwargs: Parámetros de selección de características
            
        Returns:
            Diccionario con las características seleccionadas
        """
        with self.logger.operation_trace("seleccion_caracteristicas"):
            if data_key is None:
                data_key = self.current_dataset
            
            if data_key not in self.data_cache:
                raise ValueError(f"Datos no encontrados en caché: {data_key}")
            
            self.logger.info(f"Seleccionando características: {data_key}")
            
            try:
                data = self.data_cache[data_key]['data']
                
                # Realizar feature engineering y selección
                feature_result = self.feature_engineer.select_features(
                    data, 
                    **kwargs
                )
                
                # Guardar en caché
                cache_key = f"features_{data_key}"
                self.data_cache[cache_key] = {
                    'data': feature_result['selected_data'],
                    'metadata': {
                        'original_key': data_key,
                        'feature_selection_params': kwargs,
                        'selected_features': feature_result['selected_features'],
                        'feature_importance': feature_result.get('feature_importance', {}),
                        'selected_at': datetime.now().isoformat()
                    },
                    'feature_info': feature_result
                }
                
                # Registrar en historial
                self.processing_history.append({
                    'operation': 'select_features',
                    'original_key': data_key,
                    'features_key': cache_key,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'original_features': data.shape[1],
                    'selected_features': feature_result['selected_data'].shape[1],
                    'selection_method': kwargs.get('method', 'unknown')
                })
                
                self.logger.info(f"Características seleccionadas: {feature_result['selected_data'].shape[1]}")
                
                return {
                    'data': feature_result['selected_data'],
                    'metadata': self.data_cache[cache_key]['metadata'],
                    'feature_info': feature_result,
                    'cache_key': cache_key
                }
                
            except Exception as e:
                self.logger.log_exception(e, "seleccion_caracteristicas")
                raise
    
    def split_data(self, data_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba
        
        Args:
            data_key: Clave de los datos en caché
            **kwargs: Parámetros de división
            
        Returns:
            Diccionario con los datos divididos
        """
        with self.logger.operation_trace("division_datos"):
            if data_key is None:
                data_key = self.current_dataset
            
            if data_key not in self.data_cache:
                raise ValueError(f"Datos no encontrados en caché: {data_key}")
            
            self.logger.info(f"Dividiendo datos: {data_key}")
            
            try:
                data = self.data_cache[data_key]['data']
                
                # Obtener parámetros de división
                test_size = kwargs.get('test_size', self.config.data.test_size)
                random_state = kwargs.get('random_state', self.config.data.random_state)
                stratify = kwargs.get('stratify', None)
                
                # Realizar división
                from sklearn.model_selection import train_test_split
                
                if stratify is not None and stratify in data.columns:
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=test_size, 
                        random_state=random_state,
                        stratify=data[stratify]
                    )
                else:
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                
                # Guardar en caché
                train_key = f"train_{data_key}"
                test_key = f"test_{data_key}"
                
                self.data_cache[train_key] = {
                    'data': train_data,
                    'metadata': {
                        'original_key': data_key,
                        'split_params': kwargs,
                        'split_type': 'train',
                        'split_at': datetime.now().isoformat()
                    }
                }
                
                self.data_cache[test_key] = {
                    'data': test_data,
                    'metadata': {
                        'original_key': data_key,
                        'split_params': kwargs,
                        'split_type': 'test',
                        'split_at': datetime.now().isoformat()
                    }
                }
                
                # Registrar en historial
                self.processing_history.append({
                    'operation': 'split_data',
                    'original_key': data_key,
                    'train_key': train_key,
                    'test_key': test_key,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'test_ratio': test_size
                })
                
                self.logger.info(f"Datos divididos: Train={len(train_data)}, Test={len(test_data)}")
                
                return {
                    'train_data': train_data,
                    'test_data': test_data,
                    'train_key': train_key,
                    'test_key': test_key,
                    'split_params': kwargs
                }
                
            except Exception as e:
                self.logger.log_exception(e, "division_datos")
                raise
    
    def get_data_summary(self, data_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene un resumen de los datos
        
        Args:
            data_key: Clave de los datos en caché
            
        Returns:
            Resumen de los datos
        """
        if data_key is None:
            data_key = self.current_dataset
        
        if data_key not in self.data_cache:
            raise ValueError(f"Datos no encontrados en caché: {data_key}")
        
        data = self.data_cache[data_key]['data']
        
        return {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'categorical_summary': {
                col: data[col].value_counts().to_dict() 
                for col in data.select_dtypes(include=['object', 'category']).columns
            },
            'loaded_at': self.data_cache[data_key].get('loaded_at', 'unknown'),
            'cache_key': data_key
        }
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de procesamiento
        
        Args:
            limit: Número máximo de registros a retornar
            
        Returns:
            Historial de procesamiento
        """
        return self.processing_history[-limit:] if self.processing_history else []
    
    def clear_cache(self, key: Optional[str] = None):
        """
        Limpia el caché de datos
        
        Args:
            key: Clave específica a limpiar (si None, limpia todo)
        """
        if key is None:
            self.data_cache.clear()
            self.logger.info("Caché de datos limpiado completamente")
        else:
            if key in self.data_cache:
                del self.data_cache[key]
                self.logger.info(f"Clave '{key}' eliminada del caché")
            else:
                self.logger.warning(f"Clave '{key}' no encontrada en caché")
    
    def save_cache(self, output_path: str):
        """
        Guarda el caché de datos en disco
        
        Args:
            output_path: Ruta donde guardar el caché
        """
        try:
            # Convertir DataFrames a formato serializable
            serializable_cache = {}
            for key, value in self.data_cache.items():
                if 'data' in value and isinstance(value['data'], pd.DataFrame):
                    serializable_cache[key] = {
                        **value,
                        'data': value['data'].to_dict('records')
                    }
                else:
                    serializable_cache[key] = value
            
            joblib.dump(serializable_cache, output_path)
            self.logger.info(f"Caché guardado en: {output_path}")
            
        except Exception as e:
            self.logger.log_exception(e, "guardar_cache")
            raise
    
    def load_cache(self, cache_path: str):
        """
        Carga el caché de datos desde disco
        
        Args:
            cache_path: Ruta del archivo de caché
        """
        try:
            cached_data = joblib.load(cache_path)
            
            # Convertir de vuelta a DataFrames
            for key, value in cached_data.items():
                if 'data' in value and isinstance(value['data'], list):
                    self.data_cache[key] = {
                        **value,
                        'data': pd.DataFrame(value['data'])
                    }
                else:
                    self.data_cache[key] = value
            
            self.logger.info(f"Caché cargado desde: {cache_path}")
            
        except Exception as e:
            self.logger.log_exception(e, "cargar_cache")
            raise
    
    def export_data(self, data_key: str, output_path: str, format: str = 'csv'):
        """
        Exporta datos a un archivo
        
        Args:
            data_key: Clave de los datos a exportar
            output_path: Ruta de salida
            format: Formato de exportación
        """
        if data_key not in self.data_cache:
            raise ValueError(f"Datos no encontrados en caché: {data_key}")
        
        data = self.data_cache[data_key]['data']
        
        try:
            if format.lower() == 'csv':
                data.to_csv(output_path, index=False)
            elif format.lower() == 'excel':
                data.to_excel(output_path, index=False)
            elif format.lower() == 'json':
                data.to_json(output_path, orient='records')
            elif format.lower() == 'parquet':
                data.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            self.logger.info(f"Datos exportados a: {output_path}")
            
        except Exception as e:
            self.logger.log_exception(e, "exportar_datos")
            raise
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el caché
        
        Returns:
            Información del caché
        """
        return {
            'total_entries': len(self.data_cache),
            'keys': list(self.data_cache.keys()),
            'current_dataset': self.current_dataset,
            'cache_size_mb': sum(
                len(str(value).encode('utf-8')) / (1024 * 1024) 
                for value in self.data_cache.values()
            ),
            'processing_history_count': len(self.processing_history)
        }
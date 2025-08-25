"""
Cargador de Datos - PMS 4.0.0
============================

Este módulo se encarga de cargar datos desde diferentes formatos de archivo,
incluyendo CSV, Excel, JSON, Parquet, y otros formatos comunes.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import yaml
import pickle
from datetime import datetime
import warnings

from ...controller.logger import Logger
from ...controller.config import Config


class DataLoader:
    """
    Cargador de datos para PMS 4.0.0
    
    Responsabilidades:
    - Carga de datos desde múltiples formatos
    - Detección automática de formato
    - Validación de archivos
    - Extracción de metadatos
    """
    
    def __init__(self, config: Config, logger: Logger):
        """
        Inicializa el cargador de datos
        
        Args:
            config: Configuración del sistema
            logger: Sistema de logging
        """
        self.config = config
        self.logger = logger
        
        # Formatos soportados
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.pickle': self._load_pickle,
            '.pkl': self._load_pickle,
            '.yaml': self._load_yaml,
            '.yml': self._load_yaml,
            '.txt': self._load_text,
            '.tsv': self._load_tsv
        }
        
        self.logger.info("Cargador de datos inicializado correctamente")
    
    def load_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo de datos
        
        Args:
            file_path: Ruta al archivo
            **kwargs: Parámetros adicionales para la carga
            
        Returns:
            Diccionario con los datos y metadatos
        """
        with self.logger.operation_trace("cargar_archivo"):
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            # Detectar formato
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Formato no soportado: {file_extension}")
            
            self.logger.info(f"Cargando archivo: {file_path} (formato: {file_extension})")
            
            try:
                # Cargar datos
                data = self.supported_formats[file_extension](file_path, **kwargs)
                
                # Extraer metadatos
                metadata = self._extract_metadata(file_path, data, **kwargs)
                
                self.logger.info(f"Archivo cargado exitosamente: {data.shape}")
                
                return {
                    'data': data,
                    'metadata': metadata,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'loaded_at': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.log_exception(e, f"cargar_archivo_{file_extension}")
                raise
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo CSV"""
        # Parámetros por defecto para CSV
        default_params = {
            'encoding': 'utf-8',
            'sep': ',',
            'delimiter': None,
            'header': 0,
            'index_col': None,
            'na_values': ['', 'NA', 'null', 'NULL', 'NaN', 'nan'],
            'keep_default_na': True,
            'low_memory': False
        }
        
        # Combinar parámetros por defecto con los proporcionados
        load_params = {**default_params, **kwargs}
        
        try:
            data = pd.read_csv(file_path, **load_params)
            return data
        except UnicodeDecodeError:
            # Intentar con diferentes encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    load_params['encoding'] = encoding
                    data = pd.read_csv(file_path, **load_params)
                    self.logger.info(f"Archivo cargado con encoding: {encoding}")
                    return data
                except UnicodeDecodeError:
                    continue
            raise ValueError("No se pudo determinar el encoding del archivo")
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo Excel"""
        default_params = {
            'sheet_name': 0,  # Primera hoja por defecto
            'header': 0,
            'index_col': None,
            'na_values': ['', 'NA', 'null', 'NULL', 'NaN', 'nan'],
            'keep_default_na': True
        }
        
        load_params = {**default_params, **kwargs}
        
        try:
            data = pd.read_excel(file_path, **load_params)
            return data
        except Exception as e:
            self.logger.warning(f"Error al cargar Excel: {e}")
            # Intentar con engine='openpyxl'
            try:
                load_params['engine'] = 'openpyxl'
                data = pd.read_excel(file_path, **load_params)
                return data
            except Exception as e2:
                raise ValueError(f"No se pudo cargar el archivo Excel: {e2}")
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo JSON"""
        default_params = {
            'orient': 'records',
            'lines': False,
            'encoding': 'utf-8'
        }
        
        load_params = {**default_params, **kwargs}
        
        try:
            data = pd.read_json(file_path, **load_params)
            return data
        except Exception as e:
            self.logger.warning(f"Error al cargar JSON: {e}")
            # Intentar con diferentes orientaciones
            for orient in ['records', 'split', 'index', 'columns', 'values']:
                try:
                    load_params['orient'] = orient
                    data = pd.read_json(file_path, **load_params)
                    self.logger.info(f"JSON cargado con orientación: {orient}")
                    return data
                except Exception:
                    continue
            raise ValueError("No se pudo cargar el archivo JSON")
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo Parquet"""
        try:
            data = pd.read_parquet(file_path, **kwargs)
            return data
        except Exception as e:
            raise ValueError(f"No se pudo cargar el archivo Parquet: {e}")
    
    def _load_pickle(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo Pickle"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convertir a DataFrame si no lo es
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame(data)
                elif isinstance(data, (list, tuple)):
                    data = pd.DataFrame(data)
                else:
                    raise ValueError("Contenido del pickle no es convertible a DataFrame")
            
            return data
        except Exception as e:
            raise ValueError(f"No se pudo cargar el archivo Pickle: {e}")
    
    def _load_yaml(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_dict = yaml.safe_load(f)
            
            # Convertir a DataFrame
            if isinstance(data_dict, dict):
                # Si es un diccionario simple, convertirlo a DataFrame
                if all(isinstance(v, (list, tuple)) for v in data_dict.values()):
                    data = pd.DataFrame(data_dict)
                else:
                    # Si es un diccionario anidado, intentar extraer datos
                    data = pd.json_normalize(data_dict)
            elif isinstance(data_dict, list):
                data = pd.DataFrame(data_dict)
            else:
                raise ValueError("Formato YAML no soportado")
            
            return data
        except Exception as e:
            raise ValueError(f"No se pudo cargar el archivo YAML: {e}")
    
    def _load_text(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo de texto"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Intentar parsear como datos tabulares
            data_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Intentar diferentes separadores
                    for sep in ['\t', ',', ';', '|']:
                        if sep in line:
                            data_lines.append(line.split(sep))
                            break
                    else:
                        # Si no hay separador, tratar como una sola columna
                        data_lines.append([line])
            
            if data_lines:
                data = pd.DataFrame(data_lines)
                # Intentar usar la primera fila como encabezados
                if kwargs.get('header', True):
                    data.columns = data.iloc[0]
                    data = data.iloc[1:].reset_index(drop=True)
                return data
            else:
                raise ValueError("No se encontraron datos válidos en el archivo de texto")
                
        except Exception as e:
            raise ValueError(f"No se pudo cargar el archivo de texto: {e}")
    
    def _load_tsv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Carga archivo TSV (Tab Separated Values)"""
        kwargs['sep'] = '\t'
        return self._load_csv(file_path, **kwargs)
    
    def _extract_metadata(self, file_path: Path, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Extrae metadatos del archivo y los datos
        
        Args:
            file_path: Ruta del archivo
            data: DataFrame cargado
            **kwargs: Parámetros de carga
            
        Returns:
            Diccionario con metadatos
        """
        metadata = {
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'file_size_bytes': file_path.stat().st_size,
            'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'data_shape': data.shape,
            'data_types': data.dtypes.to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'missing_values': data.isnull().sum().to_dict(),
            'unique_values_per_column': {
                col: data[col].nunique() for col in data.columns
            },
            'load_parameters': kwargs,
            'extracted_at': datetime.now().isoformat()
        }
        
        # Información adicional para columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metadata['numeric_summary'] = {
                col: {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'median': data[col].median()
                } for col in numeric_cols
            }
        
        # Información adicional para columnas categóricas
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            metadata['categorical_summary'] = {
                col: {
                    'unique_count': data[col].nunique(),
                    'most_common': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    'most_common_count': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0
                } for col in categorical_cols
            }
        
        return metadata
    
    def load_multiple_files(self, file_paths: List[str], **kwargs) -> Dict[str, Any]:
        """
        Carga múltiples archivos
        
        Args:
            file_paths: Lista de rutas de archivos
            **kwargs: Parámetros adicionales
            
        Returns:
            Diccionario con todos los datos cargados
        """
        with self.logger.operation_trace("cargar_multiples_archivos"):
            self.logger.info(f"Cargando {len(file_paths)} archivos")
            
            loaded_data = {}
            combined_data = None
            
            for file_path in file_paths:
                try:
                    result = self.load_file(file_path, **kwargs)
                    loaded_data[Path(file_path).stem] = result
                    
                    # Combinar datos si es necesario
                    if combined_data is None:
                        combined_data = result['data']
                    else:
                        # Intentar concatenar verticalmente
                        try:
                            combined_data = pd.concat([combined_data, result['data']], 
                                                    ignore_index=True, sort=False)
                        except Exception as e:
                            self.logger.warning(f"No se pudo combinar {file_path}: {e}")
                    
                except Exception as e:
                    self.logger.error(f"Error al cargar {file_path}: {e}")
                    continue
            
            return {
                'individual_files': loaded_data,
                'combined_data': combined_data,
                'total_files': len(file_paths),
                'successful_loads': len(loaded_data),
                'failed_loads': len(file_paths) - len(loaded_data)
            }
    
    def validate_file_format(self, file_path: str) -> Dict[str, Any]:
        """
        Valida el formato de un archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Información de validación
        """
        file_path = Path(file_path)
        
        validation_result = {
            'file_exists': file_path.exists(),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_extension': file_path.suffix.lower(),
            'format_supported': file_path.suffix.lower() in self.supported_formats,
            'can_read': os.access(file_path, os.R_OK) if file_path.exists() else False,
            'validation_time': datetime.now().isoformat()
        }
        
        if validation_result['file_exists'] and validation_result['format_supported']:
            try:
                # Intentar cargar una pequeña muestra para validar
                if file_path.suffix.lower() == '.csv':
                    sample = pd.read_csv(file_path, nrows=5)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    sample = pd.read_excel(file_path, nrows=5)
                elif file_path.suffix.lower() == '.json':
                    sample = pd.read_json(file_path)
                    if len(sample) > 5:
                        sample = sample.head(5)
                else:
                    sample = self.supported_formats[file_path.suffix.lower()](file_path)
                    if isinstance(sample, pd.DataFrame) and len(sample) > 5:
                        sample = sample.head(5)
                
                validation_result['sample_load_successful'] = True
                validation_result['sample_shape'] = sample.shape
                validation_result['sample_columns'] = list(sample.columns) if hasattr(sample, 'columns') else []
                
            except Exception as e:
                validation_result['sample_load_successful'] = False
                validation_result['sample_load_error'] = str(e)
        
        return validation_result
    
    def get_supported_formats(self) -> List[str]:
        """
        Obtiene la lista de formatos soportados
        
        Returns:
            Lista de extensiones soportadas
        """
        return list(self.supported_formats.keys())
    
    def create_sample_data(self, rows: int = 100, columns: int = 10, 
                          data_type: str = 'mixed') -> pd.DataFrame:
        """
        Crea datos de muestra para pruebas
        
        Args:
            rows: Número de filas
            columns: Número de columnas
            data_type: Tipo de datos ('numeric', 'categorical', 'mixed')
            
        Returns:
            DataFrame con datos de muestra
        """
        np.random.seed(42)
        
        if data_type == 'numeric':
            data = pd.DataFrame(
                np.random.randn(rows, columns),
                columns=[f'col_{i}' for i in range(columns)]
            )
        elif data_type == 'categorical':
            categories = ['A', 'B', 'C', 'D', 'E']
            data = pd.DataFrame({
                f'col_{i}': np.random.choice(categories, rows)
                for i in range(columns)
            })
        else:  # mixed
            numeric_cols = columns // 2
            categorical_cols = columns - numeric_cols
            
            numeric_data = pd.DataFrame(
                np.random.randn(rows, numeric_cols),
                columns=[f'numeric_{i}' for i in range(numeric_cols)]
            )
            
            categories = ['A', 'B', 'C', 'D', 'E']
            categorical_data = pd.DataFrame({
                f'categorical_{i}': np.random.choice(categories, rows)
                for i in range(categorical_cols)
            })
            
            data = pd.concat([numeric_data, categorical_data], axis=1)
        
        return data
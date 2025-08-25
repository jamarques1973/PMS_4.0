"""
Validador de Datos - PMS 4.0.0
=============================

Este módulo se encarga de validar la calidad y consistencia de los datos,
incluyendo verificación de tipos, valores faltantes, outliers, y otros
aspectos de calidad de datos.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime

from ...controller.logger import Logger
from ...controller.config import Config


class DataValidator:
    """
    Validador de datos para PMS 4.0.0
    
    Responsabilidades:
    - Validación de calidad de datos
    - Detección de outliers
    - Verificación de consistencia
    - Generación de reportes de validación
    """
    
    def __init__(self, config: Config, logger: Logger):
        """
        Inicializa el validador de datos
        
        Args:
            config: Configuración del sistema
            logger: Sistema de logging
        """
        self.config = config
        self.logger = logger
        
        # Configuración de validación
        self.validation_rules = {
            'missing_threshold': 0.5,  # 50% de valores faltantes máximo
            'outlier_threshold': 3.0,  # 3 desviaciones estándar para outliers
            'duplicate_threshold': 0.1,  # 10% de duplicados máximo
            'min_rows': 10,  # Mínimo número de filas
            'min_columns': 2,  # Mínimo número de columnas
            'max_columns': 1000,  # Máximo número de columnas
            'max_rows': 1000000,  # Máximo número de filas
        }
        
        # Actualizar con configuración del sistema
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'validation'):
            self.validation_rules.update(self.config.data.validation.__dict__)
        
        self.logger.info("Validador de datos inicializado correctamente")
    
    def validate_dataset(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Valida un dataset completo
        
        Args:
            data: DataFrame a validar
            **kwargs: Parámetros adicionales de validación
            
        Returns:
            Resultado de la validación
        """
        with self.logger.operation_trace("validacion_dataset"):
            self.logger.info(f"Validando dataset: {data.shape}")
            
            validation_results = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'summary': {},
                'timestamp': datetime.now().isoformat(),
                'data_shape': data.shape
            }
            
            try:
                # Validaciones básicas
                basic_validation = self._validate_basic_structure(data)
                validation_results['basic_validation'] = basic_validation
                
                if not basic_validation['is_valid']:
                    validation_results['is_valid'] = False
                    validation_results['issues'].extend(basic_validation['issues'])
                
                # Validación de tipos de datos
                type_validation = self._validate_data_types(data)
                validation_results['type_validation'] = type_validation
                
                if not type_validation['is_valid']:
                    validation_results['warnings'].extend(type_validation['warnings'])
                
                # Validación de valores faltantes
                missing_validation = self._validate_missing_values(data)
                validation_results['missing_validation'] = missing_validation
                
                if not missing_validation['is_valid']:
                    validation_results['issues'].extend(missing_validation['issues'])
                
                # Validación de duplicados
                duplicate_validation = self._validate_duplicates(data)
                validation_results['duplicate_validation'] = duplicate_validation
                
                if not duplicate_validation['is_valid']:
                    validation_results['warnings'].extend(duplicate_validation['warnings'])
                
                # Validación de outliers (solo para columnas numéricas)
                outlier_validation = self._validate_outliers(data)
                validation_results['outlier_validation'] = outlier_validation
                
                if outlier_validation['outlier_count'] > 0:
                    validation_results['warnings'].append(
                        f"Se encontraron {outlier_validation['outlier_count']} outliers"
                    )
                
                # Validación de consistencia
                consistency_validation = self._validate_consistency(data)
                validation_results['consistency_validation'] = consistency_validation
                
                if not consistency_validation['is_valid']:
                    validation_results['issues'].extend(consistency_validation['issues'])
                
                # Generar resumen
                validation_results['summary'] = self._generate_validation_summary(validation_results)
                
                # Determinar si el dataset es válido
                validation_results['is_valid'] = len(validation_results['issues']) == 0
                
                self.logger.info(f"Validación completada: {'Válido' if validation_results['is_valid'] else 'Problemas detectados'}")
                
                return validation_results
                
            except Exception as e:
                self.logger.log_exception(e, "validacion_dataset")
                validation_results['is_valid'] = False
                validation_results['issues'].append(f"Error en validación: {str(e)}")
                return validation_results
    
    def _validate_basic_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida la estructura básica del dataset"""
        issues = []
        is_valid = True
        
        # Verificar número de filas
        if len(data) < self.validation_rules['min_rows']:
            issues.append(f"Dataset tiene muy pocas filas: {len(data)} < {self.validation_rules['min_rows']}")
            is_valid = False
        
        if len(data) > self.validation_rules['max_rows']:
            issues.append(f"Dataset tiene demasiadas filas: {len(data)} > {self.validation_rules['max_rows']}")
            is_valid = False
        
        # Verificar número de columnas
        if len(data.columns) < self.validation_rules['min_columns']:
            issues.append(f"Dataset tiene muy pocas columnas: {len(data.columns)} < {self.validation_rules['min_columns']}")
            is_valid = False
        
        if len(data.columns) > self.validation_rules['max_columns']:
            issues.append(f"Dataset tiene demasiadas columnas: {len(data.columns)} > {self.validation_rules['max_columns']}")
            is_valid = False
        
        # Verificar que no esté vacío
        if data.empty:
            issues.append("Dataset está vacío")
            is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'row_count': len(data),
            'column_count': len(data.columns),
            'is_empty': data.empty
        }
    
    def _validate_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida los tipos de datos"""
        warnings = []
        is_valid = True
        
        # Verificar tipos de datos
        for col in data.columns:
            dtype = data[col].dtype
            
            # Verificar si hay mezcla de tipos en columnas
            if dtype == 'object':
                # Verificar si es realmente categórico o mixto
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.1:  # Menos del 10% de valores únicos
                    warnings.append(f"Columna '{col}' podría ser categórica pero está como object")
            
            # Verificar si hay columnas numéricas con muchos valores únicos
            elif np.issubdtype(dtype, np.number):
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio > 0.95:  # Más del 95% de valores únicos
                    warnings.append(f"Columna '{col}' tiene muchos valores únicos, podría ser un ID")
        
        return {
            'is_valid': is_valid,
            'warnings': warnings,
            'dtype_summary': data.dtypes.value_counts().to_dict(),
            'object_columns': list(data.select_dtypes(include=['object']).columns),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['category']).columns)
        }
    
    def _validate_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida los valores faltantes"""
        issues = []
        is_valid = True
        
        missing_summary = data.isnull().sum()
        missing_ratio = missing_summary / len(data)
        
        # Verificar columnas con demasiados valores faltantes
        high_missing_cols = missing_ratio[missing_ratio > self.validation_rules['missing_threshold']]
        
        if len(high_missing_cols) > 0:
            for col in high_missing_cols.index:
                issues.append(
                    f"Columna '{col}' tiene {missing_ratio[col]:.2%} de valores faltantes "
                    f"(máximo permitido: {self.validation_rules['missing_threshold']:.2%})"
                )
            is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'total_missing': missing_summary.sum(),
            'missing_ratio': missing_ratio.to_dict(),
            'columns_with_missing': list(missing_summary[missing_summary > 0].index),
            'high_missing_columns': list(high_missing_cols.index)
        }
    
    def _validate_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida duplicados"""
        warnings = []
        is_valid = True
        
        # Verificar filas duplicadas
        duplicate_count = data.duplicated().sum()
        duplicate_ratio = duplicate_count / len(data)
        
        if duplicate_ratio > self.validation_rules['duplicate_threshold']:
            warnings.append(
                f"Dataset tiene {duplicate_ratio:.2%} de filas duplicadas "
                f"(máximo recomendado: {self.validation_rules['duplicate_threshold']:.2%})"
            )
        
        # Verificar columnas con valores duplicados
        high_duplicate_cols = []
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.1:  # Menos del 10% de valores únicos
                high_duplicate_cols.append(col)
        
        if high_duplicate_cols:
            warnings.append(f"Columnas con muchos valores duplicados: {high_duplicate_cols}")
        
        return {
            'is_valid': is_valid,
            'warnings': warnings,
            'duplicate_rows': duplicate_count,
            'duplicate_ratio': duplicate_ratio,
            'high_duplicate_columns': high_duplicate_cols
        }
    
    def _validate_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida outliers en columnas numéricas"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        total_outliers = 0
        
        for col in numeric_cols:
            # Usar método IQR para detectar outliers
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': outlier_count,
                    'ratio': outlier_count / len(data),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_values': outliers[col].tolist()
                }
                total_outliers += outlier_count
        
        return {
            'outlier_count': total_outliers,
            'outlier_info': outlier_info,
            'columns_with_outliers': list(outlier_info.keys())
        }
    
    def _validate_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Valida la consistencia de los datos"""
        issues = []
        is_valid = True
        
        # Verificar consistencia de rangos en columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Verificar valores negativos en columnas que no deberían tenerlos
            if 'count' in col.lower() or 'quantity' in col.lower() or 'amount' in col.lower():
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Columna '{col}' tiene {negative_count} valores negativos")
            
            # Verificar valores cero en columnas que no deberían tenerlos
            if 'price' in col.lower() or 'cost' in col.lower() or 'value' in col.lower():
                zero_count = (data[col] == 0).sum()
                if zero_count > len(data) * 0.5:  # Más del 50% son ceros
                    issues.append(f"Columna '{col}' tiene muchos valores cero ({zero_count})")
        
        # Verificar consistencia de fechas
        date_cols = []
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(data[col])
                    date_cols.append(col)
                except:
                    pass
        
        for col in date_cols:
            try:
                dates = pd.to_datetime(data[col])
                # Verificar fechas futuras
                future_dates = dates > pd.Timestamp.now()
                if future_dates.sum() > 0:
                    issues.append(f"Columna '{col}' tiene fechas futuras")
            except:
                issues.append(f"Columna '{col}' tiene formato de fecha inconsistente")
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'numeric_columns_checked': len(numeric_cols),
            'date_columns_found': len(date_cols)
        }
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un resumen de la validación"""
        return {
            'total_issues': len(validation_results['issues']),
            'total_warnings': len(validation_results['warnings']),
            'validation_score': self._calculate_validation_score(validation_results),
            'critical_issues': [issue for issue in validation_results['issues'] 
                              if 'error' in issue.lower() or 'vacío' in issue.lower()],
            'recommendations': self._generate_recommendations(validation_results)
        }
    
    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calcula un score de validación (0-100)"""
        score = 100.0
        
        # Penalizar por issues críticos
        score -= len(validation_results['issues']) * 10
        
        # Penalizar por warnings
        score -= len(validation_results['warnings']) * 2
        
        # Penalizar por problemas de estructura
        if 'basic_validation' in validation_results:
            basic = validation_results['basic_validation']
            if not basic['is_valid']:
                score -= 20
        
        # Penalizar por valores faltantes
        if 'missing_validation' in validation_results:
            missing = validation_results['missing_validation']
            if not missing['is_valid']:
                score -= 15
        
        return max(0.0, score)
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en los problemas encontrados"""
        recommendations = []
        
        if 'basic_validation' in validation_results:
            basic = validation_results['basic_validation']
            if basic['row_count'] < self.validation_rules['min_rows']:
                recommendations.append("Considerar obtener más datos para mejorar la robustez del modelo")
        
        if 'missing_validation' in validation_results:
            missing = validation_results['missing_validation']
            if missing['high_missing_columns']:
                recommendations.append("Considerar eliminar o imputar columnas con muchos valores faltantes")
        
        if 'duplicate_validation' in validation_results:
            duplicate = validation_results['duplicate_validation']
            if duplicate['duplicate_ratio'] > 0.05:
                recommendations.append("Considerar eliminar filas duplicadas")
        
        if 'outlier_validation' in validation_results:
            outlier = validation_results['outlier_validation']
            if outlier['outlier_count'] > 0:
                recommendations.append("Revisar outliers para determinar si son errores o valores válidos")
        
        return recommendations
    
    def validate_preprocessed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida datos preprocesados
        
        Args:
            data: DataFrame preprocesado
            
        Returns:
            Resultado de la validación
        """
        with self.logger.operation_trace("validacion_datos_preprocesados"):
            self.logger.info(f"Validando datos preprocesados: {data.shape}")
            
            # Validación básica
            validation_result = self.validate_dataset(data)
            
            # Validaciones adicionales para datos preprocesados
            additional_checks = {
                'scaled_check': self._check_scaling(data),
                'encoded_check': self._check_encoding(data),
                'feature_check': self._check_feature_quality(data)
            }
            
            validation_result['preprocessing_checks'] = additional_checks
            
            return validation_result
    
    def _check_scaling(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica si los datos están escalados correctamente"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        scaling_info = {}
        for col in numeric_cols:
            mean = data[col].mean()
            std = data[col].std()
            
            scaling_info[col] = {
                'mean': mean,
                'std': std,
                'is_centered': abs(mean) < 0.1,  # Cerca de cero
                'is_scaled': 0.5 < std < 2.0  # Desviación estándar razonable
            }
        
        return {
            'scaling_info': scaling_info,
            'properly_scaled_columns': [
                col for col, info in scaling_info.items() 
                if info['is_centered'] and info['is_scaled']
            ]
        }
    
    def _check_encoding(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica si las variables categóricas están codificadas correctamente"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        encoding_info = {}
        for col in categorical_cols:
            unique_count = data[col].nunique()
            encoding_info[col] = {
                'unique_count': unique_count,
                'needs_encoding': unique_count > 2,  # Más de 2 categorías
                'encoding_type': 'one_hot' if unique_count <= 10 else 'label'
            }
        
        return {
            'encoding_info': encoding_info,
            'columns_needing_encoding': [
                col for col, info in encoding_info.items() 
                if info['needs_encoding']
            ]
        }
    
    def _check_feature_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verifica la calidad de las características"""
        quality_info = {}
        
        for col in data.columns:
            # Variabilidad
            if data[col].dtype in ['object', 'category']:
                variability = data[col].nunique() / len(data)
            else:
                variability = data[col].std() / (data[col].max() - data[col].min()) if data[col].max() != data[col].min() else 0
            
            # Correlación con otras variables (simplificado)
            correlations = []
            if data[col].dtype in [np.number]:
                for other_col in data.select_dtypes(include=[np.number]).columns:
                    if col != other_col:
                        corr = data[col].corr(data[other_col])
                        if abs(corr) > 0.8:
                            correlations.append((other_col, corr))
            
            quality_info[col] = {
                'variability': variability,
                'high_correlations': correlations,
                'is_useful': variability > 0.01 and len(correlations) < 3
            }
        
        return {
            'quality_info': quality_info,
            'useful_features': [
                col for col, info in quality_info.items() 
                if info['is_useful']
            ],
            'low_variability_features': [
                col for col, info in quality_info.items() 
                if info['variability'] <= 0.01
            ]
        }
    
    def get_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Genera un reporte de validación en formato texto
        
        Args:
            validation_results: Resultados de validación
            
        Returns:
            Reporte de validación
        """
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE VALIDACIÓN DE DATOS")
        report.append("=" * 60)
        report.append(f"Fecha: {validation_results['timestamp']}")
        report.append(f"Dataset: {validation_results['data_shape']}")
        report.append(f"Estado: {'VÁLIDO' if validation_results['is_valid'] else 'PROBLEMAS DETECTADOS'}")
        report.append(f"Score: {validation_results['summary']['validation_score']:.1f}/100")
        report.append("")
        
        # Issues críticos
        if validation_results['issues']:
            report.append("PROBLEMAS CRÍTICOS:")
            report.append("-" * 30)
            for issue in validation_results['issues']:
                report.append(f"❌ {issue}")
            report.append("")
        
        # Warnings
        if validation_results['warnings']:
            report.append("ADVERTENCIAS:")
            report.append("-" * 20)
            for warning in validation_results['warnings']:
                report.append(f"⚠️ {warning}")
            report.append("")
        
        # Recomendaciones
        if validation_results['summary']['recommendations']:
            report.append("RECOMENDACIONES:")
            report.append("-" * 20)
            for rec in validation_results['summary']['recommendations']:
                report.append(f"💡 {rec}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
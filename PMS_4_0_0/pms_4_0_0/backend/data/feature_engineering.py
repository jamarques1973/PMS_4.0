"""
Ingeniería de Características - PMS 4.0.0
=======================================

Este módulo se encarga de la creación, transformación y selección de características,
incluyendo feature engineering avanzado y selección automática de features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import boruta

from ...controller.logger import Logger
from ...controller.config import Config


class FeatureEngineer:
    """
    Ingeniero de características para PMS 4.0.0
    
    Responsabilidades:
    - Creación de nuevas características
    - Transformación de características existentes
    - Selección automática de características
    - Análisis de importancia de características
    """
    
    def __init__(self, config: Config, logger: Logger):
        """
        Inicializa el ingeniero de características
        
        Args:
            config: Configuración del sistema
            logger: Sistema de logging
        """
        self.config = config
        self.logger = logger
        
        # Configuración de feature engineering
        self.feature_config = {
            'selection_method': 'auto',  # 'auto', 'kbest', 'rfe', 'boruta', 'lasso', 'none'
            'n_features': 'auto',  # número de características a seleccionar
            'polynomial_features': False,  # crear características polinomiales
            'polynomial_degree': 2,  # grado de características polinomiales
            'interaction_features': False,  # crear características de interacción
            'temporal_features': False,  # crear características temporales
            'statistical_features': False,  # crear características estadísticas
            'domain_features': False,  # crear características específicas del dominio
            'correlation_threshold': 0.95,  # umbral de correlación para eliminar características
            'variance_threshold': 0.01,  # umbral de varianza para eliminar características
            'mutual_info_threshold': 0.01  # umbral de información mutua
        }
        
        # Actualizar con configuración del sistema
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'feature_engineering'):
            self.feature_config.update(self.config.data.feature_engineering.__dict__)
        
        # Fitted selectors
        self.fitted_selectors = {}
        
        self.logger.info("Ingeniero de características inicializado correctamente")
    
    def create_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Crea nuevas características
        
        Args:
            data: DataFrame original
            **kwargs: Parámetros adicionales
            
        Returns:
            DataFrame con nuevas características
        """
        with self.logger.operation_trace("creacion_caracteristicas"):
            self.logger.info(f"Creando características para dataset: {data.shape}")
            
            # Combinar configuración con parámetros adicionales
            config = {**self.feature_config, **kwargs}
            
            try:
                # Crear copia para no modificar el original
                feature_data = data.copy()
                
                # 1. Características polinomiales
                if config.get('polynomial_features', False):
                    feature_data = self._create_polynomial_features(feature_data, config)
                
                # 2. Características de interacción
                if config.get('interaction_features', False):
                    feature_data = self._create_interaction_features(feature_data, config)
                
                # 3. Características temporales
                if config.get('temporal_features', False):
                    feature_data = self._create_temporal_features(feature_data, config)
                
                # 4. Características estadísticas
                if config.get('statistical_features', False):
                    feature_data = self._create_statistical_features(feature_data, config)
                
                # 5. Características específicas del dominio
                if config.get('domain_features', False):
                    feature_data = self._create_domain_features(feature_data, config)
                
                self.logger.info(f"Características creadas: {feature_data.shape}")
                
                return feature_data
                
            except Exception as e:
                self.logger.log_exception(e, "creacion_caracteristicas")
                raise
    
    def _create_polynomial_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Crea características polinomiales"""
        degree = config.get('polynomial_degree', 2)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
        
        # Limitar el número de características para evitar explosión combinatoria
        if len(numeric_cols) > 10:
            # Seleccionar las columnas más importantes
            correlations = data[numeric_cols].corr().abs().mean()
            top_cols = correlations.nlargest(10).index
            numeric_cols = top_cols
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(data[numeric_cols])
        
        # Crear nombres de características
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Crear DataFrame con características polinomiales
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
        
        # Eliminar características originales y agregar las nuevas
        result_data = data.drop(columns=numeric_cols)
        result_data = pd.concat([result_data, poly_df], axis=1)
        
        self.logger.info(f"Características polinomiales creadas: {len(feature_names)} nuevas características")
        
        return result_data
    
    def _create_interaction_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Crea características de interacción"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return data
        
        interaction_features = {}
        
        # Crear interacciones entre pares de características
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplicación
                interaction_features[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                
                # División (evitar división por cero)
                if (data[col2] != 0).all():
                    interaction_features[f'{col1}_div_{col2}'] = data[col1] / data[col2]
                
                # Suma
                interaction_features[f'{col1}_plus_{col2}'] = data[col1] + data[col2]
                
                # Resta
                interaction_features[f'{col1}_minus_{col2}'] = data[col1] - data[col2]
        
        # Crear DataFrame con características de interacción
        interaction_df = pd.DataFrame(interaction_features, index=data.index)
        
        # Concatenar con datos originales
        result_data = pd.concat([data, interaction_df], axis=1)
        
        self.logger.info(f"Características de interacción creadas: {len(interaction_features)} nuevas características")
        
        return result_data
    
    def _create_temporal_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Crea características temporales"""
        temporal_features = {}
        
        # Buscar columnas que podrían ser fechas
        date_cols = []
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                try:
                    pd.to_datetime(data[col])
                    date_cols.append(col)
                except:
                    pass
        
        for col in date_cols:
            try:
                dates = pd.to_datetime(data[col])
                
                # Extraer componentes temporales
                temporal_features[f'{col}_year'] = dates.dt.year
                temporal_features[f'{col}_month'] = dates.dt.month
                temporal_features[f'{col}_day'] = dates.dt.day
                temporal_features[f'{col}_dayofweek'] = dates.dt.dayofweek
                temporal_features[f'{col}_quarter'] = dates.dt.quarter
                temporal_features[f'{col}_is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
                
                # Características cíclicas
                temporal_features[f'{col}_month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
                temporal_features[f'{col}_month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
                temporal_features[f'{col}_day_sin'] = np.sin(2 * np.pi * dates.dt.day / 31)
                temporal_features[f'{col}_day_cos'] = np.cos(2 * np.pi * dates.dt.day / 31)
                
            except Exception as e:
                self.logger.warning(f"No se pudieron crear características temporales para {col}: {e}")
        
        if temporal_features:
            temporal_df = pd.DataFrame(temporal_features, index=data.index)
            result_data = pd.concat([data, temporal_df], axis=1)
            
            self.logger.info(f"Características temporales creadas: {len(temporal_features)} nuevas características")
            return result_data
        
        return data
    
    def _create_statistical_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Crea características estadísticas"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data
        
        statistical_features = {}
        
        # Características de ventana deslizante
        window_sizes = [3, 5, 7]
        
        for col in numeric_cols:
            for window in window_sizes:
                if len(data) >= window:
                    # Media móvil
                    statistical_features[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window, min_periods=1).mean()
                    
                    # Desviación estándar móvil
                    statistical_features[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window, min_periods=1).std()
                    
                    # Mediana móvil
                    statistical_features[f'{col}_rolling_median_{window}'] = data[col].rolling(window=window, min_periods=1).median()
                    
                    # Máximo móvil
                    statistical_features[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window, min_periods=1).max()
                    
                    # Mínimo móvil
                    statistical_features[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window, min_periods=1).min()
        
        # Características de percentiles
        for col in numeric_cols:
            statistical_features[f'{col}_percentile_25'] = data[col].quantile(0.25)
            statistical_features[f'{col}_percentile_75'] = data[col].quantile(0.75)
            statistical_features[f'{col}_iqr'] = data[col].quantile(0.75) - data[col].quantile(0.25)
        
        if statistical_features:
            statistical_df = pd.DataFrame(statistical_features, index=data.index)
            result_data = pd.concat([data, statistical_df], axis=1)
            
            self.logger.info(f"Características estadísticas creadas: {len(statistical_features)} nuevas características")
            return result_data
        
        return data
    
    def _create_domain_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Crea características específicas del dominio"""
        domain_features = {}
        
        # Ejemplos de características específicas del dominio
        # Estas se pueden personalizar según el contexto del problema
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Características de logaritmo
            if (data[col] > 0).all():
                domain_features[f'{col}_log'] = np.log(data[col])
                domain_features[f'{col}_log1p'] = np.log1p(data[col])
            
            # Características de raíz cuadrada
            if (data[col] >= 0).all():
                domain_features[f'{col}_sqrt'] = np.sqrt(data[col])
            
            # Características de cuadrado
            domain_features[f'{col}_squared'] = data[col] ** 2
            
            # Características de cubo
            domain_features[f'{col}_cubed'] = data[col] ** 3
            
            # Características de recíproco
            if (data[col] != 0).all():
                domain_features[f'{col}_reciprocal'] = 1 / data[col]
        
        if domain_features:
            domain_df = pd.DataFrame(domain_features, index=data.index)
            result_data = pd.concat([data, domain_df], axis=1)
            
            self.logger.info(f"Características de dominio creadas: {len(domain_features)} nuevas características")
            return result_data
        
        return data
    
    def select_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None, 
                       **kwargs) -> Dict[str, Any]:
        """
        Selecciona características relevantes
        
        Args:
            data: DataFrame con características
            target: Serie objetivo (opcional)
            **kwargs: Parámetros adicionales
            
        Returns:
            Diccionario con características seleccionadas y metadatos
        """
        with self.logger.operation_trace("seleccion_caracteristicas"):
            self.logger.info(f"Seleccionando características de dataset: {data.shape}")
            
            # Combinar configuración con parámetros adicionales
            config = {**self.feature_config, **kwargs}
            
            try:
                method = config.get('selection_method', 'auto')
                n_features = config.get('n_features', 'auto')
                
                # Determinar número de características automáticamente
                if n_features == 'auto':
                    n_features = min(50, len(data.columns) // 2)
                
                # Seleccionar características
                if method == 'auto':
                    selected_data, selection_info = self._auto_feature_selection(data, target, n_features, config)
                elif method == 'kbest':
                    selected_data, selection_info = self._kbest_selection(data, target, n_features, config)
                elif method == 'rfe':
                    selected_data, selection_info = self._rfe_selection(data, target, n_features, config)
                elif method == 'boruta':
                    selected_data, selection_info = self._boruta_selection(data, target, config)
                elif method == 'lasso':
                    selected_data, selection_info = self._lasso_selection(data, target, config)
                elif method == 'correlation':
                    selected_data, selection_info = self._correlation_selection(data, config)
                else:
                    # Sin selección
                    selected_data = data
                    selection_info = {
                        'method': 'none',
                        'selected_features': list(data.columns),
                        'feature_importance': {},
                        'selection_score': 1.0
                    }
                
                self.logger.info(f"Características seleccionadas: {selected_data.shape}")
                
                return {
                    'selected_data': selected_data,
                    'selected_features': list(selected_data.columns),
                    'selection_info': selection_info,
                    'original_shape': data.shape,
                    'reduction_ratio': len(selected_data.columns) / len(data.columns)
                }
                
            except Exception as e:
                self.logger.log_exception(e, "seleccion_caracteristicas")
                raise
    
    def _auto_feature_selection(self, data: pd.DataFrame, target: pd.Series, 
                               n_features: int, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selección automática de características"""
        # Intentar diferentes métodos y seleccionar el mejor
        methods = ['kbest', 'rfe', 'correlation']
        best_score = 0
        best_result = None
        best_method = None
        
        for method in methods:
            try:
                if method == 'kbest':
                    selected_data, info = self._kbest_selection(data, target, n_features, config)
                elif method == 'rfe':
                    selected_data, info = self._rfe_selection(data, target, n_features, config)
                elif method == 'correlation':
                    selected_data, info = self._correlation_selection(data, config)
                
                if info.get('selection_score', 0) > best_score:
                    best_score = info.get('selection_score', 0)
                    best_result = (selected_data, info)
                    best_method = method
                    
            except Exception as e:
                self.logger.warning(f"Método {method} falló: {e}")
                continue
        
        if best_result is None:
            # Fallback: sin selección
            return data, {
                'method': 'none',
                'selected_features': list(data.columns),
                'feature_importance': {},
                'selection_score': 0.0
            }
        
        return best_result
    
    def _kbest_selection(self, data: pd.DataFrame, target: pd.Series, 
                        n_features: int, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selección basada en estadísticas F"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data, {
                'method': 'kbest',
                'selected_features': list(data.columns),
                'feature_importance': {},
                'selection_score': 0.0
            }
        
        # Determinar si es regresión o clasificación
        if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
            # Clasificación
            selector = SelectKBest(score_func=f_classif, k=min(n_features, len(numeric_cols)))
        else:
            # Regresión
            selector = SelectKBest(score_func=f_regression, k=min(n_features, len(numeric_cols)))
        
        # Ajustar selector
        selector.fit(data[numeric_cols], target)
        
        # Obtener características seleccionadas
        selected_cols = numeric_cols[selector.get_support()]
        
        # Crear DataFrame con características seleccionadas
        selected_data = data[selected_cols].copy()
        
        # Agregar columnas no numéricas de vuelta
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        # Guardar selector
        self.fitted_selectors['kbest'] = selector
        
        # Información de selección
        feature_scores = dict(zip(numeric_cols, selector.scores_))
        
        return selected_data, {
            'method': 'kbest',
            'selected_features': list(selected_data.columns),
            'feature_importance': feature_scores,
            'selection_score': selector.scores_.mean() if len(selector.scores_) > 0 else 0.0,
            'n_selected': len(selected_cols)
        }
    
    def _rfe_selection(self, data: pd.DataFrame, target: pd.Series, 
                      n_features: int, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selección recursiva de características"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data, {
                'method': 'rfe',
                'selected_features': list(data.columns),
                'feature_importance': {},
                'selection_score': 0.0
            }
        
        # Determinar si es regresión o clasificación
        if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
            # Clasificación
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Regresión
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Crear selector RFE
        selector = RFE(estimator=estimator, n_features_to_select=min(n_features, len(numeric_cols)))
        
        # Ajustar selector
        selector.fit(data[numeric_cols], target)
        
        # Obtener características seleccionadas
        selected_cols = numeric_cols[selector.support_]
        
        # Crear DataFrame con características seleccionadas
        selected_data = data[selected_cols].copy()
        
        # Agregar columnas no numéricas de vuelta
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        # Guardar selector
        self.fitted_selectors['rfe'] = selector
        
        # Información de selección
        feature_ranking = dict(zip(numeric_cols, selector.ranking_))
        
        return selected_data, {
            'method': 'rfe',
            'selected_features': list(selected_data.columns),
            'feature_importance': feature_ranking,
            'selection_score': 1.0 - (selector.ranking_.mean() / len(numeric_cols)),
            'n_selected': len(selected_cols)
        }
    
    def _boruta_selection(self, data: pd.DataFrame, target: pd.Series, 
                         config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selección usando Boruta"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data, {
                'method': 'boruta',
                'selected_features': list(data.columns),
                'feature_importance': {},
                'selection_score': 0.0
            }
        
        # Determinar si es regresión o clasificación
        if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
            # Clasificación
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            # Regresión
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Crear selector Boruta
        selector = boruta.BorutaPy(estimator, random_state=42, verbose=0)
        
        # Ajustar selector
        selector.fit(data[numeric_cols].values, target.values)
        
        # Obtener características seleccionadas
        selected_cols = numeric_cols[selector.support_]
        
        # Crear DataFrame con características seleccionadas
        selected_data = data[selected_cols].copy()
        
        # Agregar columnas no numéricas de vuelta
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        # Guardar selector
        self.fitted_selectors['boruta'] = selector
        
        # Información de selección
        feature_importance = dict(zip(numeric_cols, selector.ranking_))
        
        return selected_data, {
            'method': 'boruta',
            'selected_features': list(selected_data.columns),
            'feature_importance': feature_importance,
            'selection_score': 1.0 - (selector.ranking_.mean() / len(numeric_cols)),
            'n_selected': len(selected_cols)
        }
    
    def _lasso_selection(self, data: pd.DataFrame, target: pd.Series, 
                        config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selección usando Lasso"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return data, {
                'method': 'lasso',
                'selected_features': list(data.columns),
                'feature_importance': {},
                'selection_score': 0.0
            }
        
        # Crear selector basado en Lasso
        selector = SelectFromModel(Lasso(alpha=0.01, random_state=42))
        
        # Ajustar selector
        selector.fit(data[numeric_cols], target)
        
        # Obtener características seleccionadas
        selected_cols = numeric_cols[selector.get_support()]
        
        # Crear DataFrame con características seleccionadas
        selected_data = data[selected_cols].copy()
        
        # Agregar columnas no numéricas de vuelta
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        # Guardar selector
        self.fitted_selectors['lasso'] = selector
        
        # Información de selección
        feature_importance = dict(zip(numeric_cols, np.abs(selector.estimator_.coef_)))
        
        return selected_data, {
            'method': 'lasso',
            'selected_features': list(selected_data.columns),
            'feature_importance': feature_importance,
            'selection_score': np.mean(np.abs(selector.estimator_.coef_)) if len(selector.estimator_.coef_) > 0 else 0.0,
            'n_selected': len(selected_cols)
        }
    
    def _correlation_selection(self, data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selección basada en correlación"""
        threshold = config.get('correlation_threshold', 0.95)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return data, {
                'method': 'correlation',
                'selected_features': list(data.columns),
                'feature_importance': {},
                'selection_score': 1.0
            }
        
        # Calcular matriz de correlación
        corr_matrix = data[numeric_cols].corr().abs()
        
        # Encontrar características altamente correlacionadas
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Encontrar características a eliminar
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Eliminar características altamente correlacionadas
        selected_cols = [col for col in numeric_cols if col not in to_drop]
        
        # Crear DataFrame con características seleccionadas
        selected_data = data[selected_cols].copy()
        
        # Agregar columnas no numéricas de vuelta
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
        
        # Información de selección
        feature_importance = {col: 1.0 for col in selected_cols}
        
        return selected_data, {
            'method': 'correlation',
            'selected_features': list(selected_data.columns),
            'feature_importance': feature_importance,
            'selection_score': 1.0 - (len(to_drop) / len(numeric_cols)),
            'n_selected': len(selected_cols),
            'dropped_features': to_drop
        }
    
    def get_feature_importance(self, data: pd.DataFrame, target: pd.Series, 
                             method: str = 'auto') -> Dict[str, float]:
        """
        Obtiene la importancia de las características
        
        Args:
            data: DataFrame con características
            target: Serie objetivo
            method: Método para calcular importancia
            
        Returns:
            Diccionario con importancia de características
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        if method == 'auto':
            # Usar Random Forest para importancia
            if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == 'mutual_info':
            # Usar información mutua
            if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
                importance = mutual_info_classif(data[numeric_cols], target, random_state=42)
            else:
                importance = mutual_info_regression(data[numeric_cols], target, random_state=42)
            
            return dict(zip(numeric_cols, importance))
        else:
            # Usar Random Forest
            if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Ajustar modelo
        model.fit(data[numeric_cols], target)
        
        # Obtener importancia
        importance = model.feature_importances_
        
        return dict(zip(numeric_cols, importance))
    
    def save_selectors(self, file_path: str):
        """
        Guarda los selectores ajustados
        
        Args:
            file_path: Ruta donde guardar los selectores
        """
        import joblib
        
        try:
            joblib.dump(self.fitted_selectors, file_path)
            self.logger.info(f"Selectores guardados en: {file_path}")
        except Exception as e:
            self.logger.log_exception(e, "guardar_selectores")
            raise
    
    def load_selectors(self, file_path: str):
        """
        Carga selectores previamente guardados
        
        Args:
            file_path: Ruta de los selectores
        """
        import joblib
        
        try:
            self.fitted_selectors = joblib.load(file_path)
            self.logger.info(f"Selectores cargados desde: {file_path}")
        except Exception as e:
            self.logger.log_exception(e, "cargar_selectores")
            raise
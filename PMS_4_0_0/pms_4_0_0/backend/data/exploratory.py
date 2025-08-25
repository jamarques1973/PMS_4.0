"""
Análisis Exploratorio - PMS 4.0.0
================================

Este módulo se encarga del análisis exploratorio de datos, incluyendo
estadísticas descriptivas, visualizaciones, detección de patrones
y análisis de correlaciones.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...controller.logger import Logger
from ...controller.config import Config


class ExploratoryAnalyzer:
    """
    Analizador exploratorio para PMS 4.0.0
    
    Responsabilidades:
    - Análisis estadístico descriptivo
    - Visualizaciones de datos
    - Detección de patrones y anomalías
    - Análisis de correlaciones
    - Reducción de dimensionalidad
    """
    
    def __init__(self, config: Config, logger: Logger):
        """
        Inicializa el analizador exploratorio
        
        Args:
            config: Configuración del sistema
            logger: Sistema de logging
        """
        self.config = config
        self.logger = logger
        
        # Configuración de análisis exploratorio
        self.analysis_config = {
            'correlation_threshold': 0.7,  # umbral para correlaciones significativas
            'outlier_threshold': 3.0,  # desviaciones estándar para outliers
            'pca_components': 0.95,  # varianza explicada para PCA
            'tsne_perplexity': 30,  # perplexidad para t-SNE
            'plot_style': 'seaborn',  # estilo de gráficos
            'figure_size': (12, 8),  # tamaño de figuras
            'dpi': 300,  # resolución de gráficos
            'save_plots': True,  # guardar gráficos automáticamente
            'plot_format': 'png'  # formato de gráficos
        }
        
        # Actualizar con configuración del sistema
        if hasattr(self.config, 'data') and hasattr(self.config.data, 'exploratory'):
            self.analysis_config.update(self.config.data.exploratory.__dict__)
        
        # Configurar estilo de gráficos
        self._setup_plotting_style()
        
        self.logger.info("Analizador exploratorio inicializado correctamente")
    
    def _setup_plotting_style(self):
        """Configura el estilo de gráficos"""
        try:
            if self.analysis_config['plot_style'] == 'seaborn':
                sns.set_style("whitegrid")
                sns.set_palette("husl")
            elif self.analysis_config['plot_style'] == 'matplotlib':
                plt.style.use('default')
            
            # Configurar tamaño de fuente
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            
        except Exception as e:
            self.logger.warning(f"No se pudo configurar el estilo de gráficos: {e}")
    
    def analyze_dataset(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Realiza análisis exploratorio completo del dataset
        
        Args:
            data: DataFrame a analizar
            **kwargs: Parámetros adicionales
            
        Returns:
            Diccionario con resultados del análisis
        """
        with self.logger.operation_trace("analisis_exploratorio"):
            self.logger.info(f"Analizando dataset: {data.shape}")
            
            # Combinar configuración con parámetros adicionales
            config = {**self.analysis_config, **kwargs}
            
            try:
                analysis_results = {
                    'basic_stats': self._basic_statistics(data),
                    'missing_analysis': self._missing_value_analysis(data),
                    'correlation_analysis': self._correlation_analysis(data, config),
                    'distribution_analysis': self._distribution_analysis(data),
                    'outlier_analysis': self._outlier_analysis(data, config),
                    'dimensionality_analysis': self._dimensionality_analysis(data, config),
                    'summary': {},
                    'timestamp': datetime.now().isoformat(),
                    'data_shape': data.shape
                }
                
                # Generar resumen
                analysis_results['summary'] = self._generate_analysis_summary(analysis_results)
                
                self.logger.info("Análisis exploratorio completado")
                
                return analysis_results
                
            except Exception as e:
                self.logger.log_exception(e, "analisis_exploratorio")
                raise
    
    def _basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estadísticas básicas"""
        stats_info = {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtypes': data.dtypes.value_counts().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Estadísticas para columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_info['numeric_summary'] = data[numeric_cols].describe().to_dict()
        
        # Estadísticas para columnas categóricas
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_stats = {}
            for col in categorical_cols:
                categorical_stats[col] = {
                    'unique_count': data[col].nunique(),
                    'most_common': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    'most_common_count': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0,
                    'top_values': data[col].value_counts().head(5).to_dict()
                }
            stats_info['categorical_summary'] = categorical_stats
        
        return stats_info
    
    def _missing_value_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza valores faltantes"""
        missing_info = {
            'total_missing': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            'missing_by_column': data.isnull().sum().to_dict(),
            'missing_percentage_by_column': (data.isnull().sum() / len(data) * 100).to_dict(),
            'columns_with_missing': list(data.columns[data.isnull().any()]),
            'rows_with_missing': data.isnull().any(axis=1).sum(),
            'complete_rows': data.dropna().shape[0]
        }
        
        return missing_info
    
    def _correlation_analysis(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza correlaciones entre variables"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'correlation_matrix': {},
                'high_correlations': [],
                'correlation_summary': {}
            }
        
        # Calcular matriz de correlación
        corr_matrix = data[numeric_cols].corr()
        
        # Encontrar correlaciones altas
        threshold = config.get('correlation_threshold', 0.7)
        high_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Ordenar por valor absoluto de correlación
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        correlation_info = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'correlation_summary': {
                'mean_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                'max_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                'min_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min(),
                'high_correlation_pairs': len(high_correlations)
            }
        }
        
        return correlation_info
    
    def _distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza distribuciones de variables"""
        distribution_info = {
            'numeric_distributions': {},
            'categorical_distributions': {},
            'skewness': {},
            'kurtosis': {}
        }
        
        # Análisis de distribuciones numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            distribution_info['numeric_distributions'][col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75),
                'iqr': data[col].quantile(0.75) - data[col].quantile(0.25)
            }
            
            # Skewness y kurtosis
            distribution_info['skewness'][col] = data[col].skew()
            distribution_info['kurtosis'][col] = data[col].kurtosis()
        
        # Análisis de distribuciones categóricas
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            distribution_info['categorical_distributions'][col] = {
                'unique_values': value_counts.to_dict(),
                'entropy': self._calculate_entropy(value_counts),
                'gini_impurity': self._calculate_gini_impurity(value_counts),
                'most_common_ratio': value_counts.iloc[0] / len(data) if len(value_counts) > 0 else 0
            }
        
        return distribution_info
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calcula la entropía de una distribución"""
        probabilities = value_counts / value_counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_gini_impurity(self, value_counts: pd.Series) -> float:
        """Calcula la impureza de Gini de una distribución"""
        probabilities = value_counts / value_counts.sum()
        return 1 - np.sum(probabilities ** 2)
    
    def _outlier_analysis(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza outliers en variables numéricas"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        outlier_info = {
            'outliers_by_column': {},
            'total_outliers': 0,
            'outlier_percentage': 0,
            'outlier_methods': {}
        }
        
        threshold = config.get('outlier_threshold', 3.0)
        
        for col in numeric_cols:
            # Método Z-score
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            z_outliers = z_scores > threshold
            
            # Método IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
            
            outlier_info['outliers_by_column'][col] = {
                'z_score_outliers': z_outliers.sum(),
                'iqr_outliers': iqr_outliers.sum(),
                'z_score_percentage': (z_outliers.sum() / len(data[col].dropna())) * 100,
                'iqr_percentage': (iqr_outliers.sum() / len(data[col])) * 100
            }
            
            outlier_info['total_outliers'] += z_outliers.sum()
        
        outlier_info['outlier_percentage'] = (outlier_info['total_outliers'] / (len(numeric_cols) * len(data))) * 100
        
        return outlier_info
    
    def _dimensionality_analysis(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la dimensionalidad de los datos"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        dimensionality_info = {
            'feature_redundancy': {},
            'pca_analysis': {},
            'tsne_analysis': {}
        }
        
        if len(numeric_cols) > 1:
            # Análisis de redundancia de características
            corr_matrix = data[numeric_cols].corr().abs()
            redundancy_scores = {}
            
            for col in numeric_cols:
                # Calcular cuántas características están altamente correlacionadas con esta
                high_corr_count = (corr_matrix[col] > 0.8).sum() - 1  # -1 para excluir la propia característica
                redundancy_scores[col] = high_corr_count / (len(numeric_cols) - 1)
            
            dimensionality_info['feature_redundancy'] = redundancy_scores
            
            # Análisis PCA
            try:
                pca = PCA()
                pca.fit(data[numeric_cols])
                
                # Calcular número de componentes para explicar varianza
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                
                target_variance = config.get('pca_components', 0.95)
                n_components = np.argmax(cumulative_variance >= target_variance) + 1
                
                dimensionality_info['pca_analysis'] = {
                    'n_components_for_95_variance': n_components,
                    'explained_variance_ratio': explained_variance_ratio.tolist(),
                    'cumulative_variance': cumulative_variance.tolist(),
                    'total_variance_explained': cumulative_variance[-1]
                }
                
            except Exception as e:
                self.logger.warning(f"Error en análisis PCA: {e}")
        
        return dimensionality_info
    
    def _generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un resumen del análisis"""
        summary = {
            'data_quality_score': self._calculate_data_quality_score(analysis_results),
            'key_insights': self._extract_key_insights(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results),
            'analysis_completeness': len(analysis_results) - 2  # Excluir summary y timestamp
        }
        
        return summary
    
    def _calculate_data_quality_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calcula un score de calidad de datos (0-100)"""
        score = 100.0
        
        # Penalizar por valores faltantes
        missing_analysis = analysis_results.get('missing_analysis', {})
        missing_percentage = missing_analysis.get('missing_percentage', 0)
        score -= missing_percentage * 0.5  # 0.5 puntos por cada 1% de valores faltantes
        
        # Penalizar por outliers
        outlier_analysis = analysis_results.get('outlier_analysis', {})
        outlier_percentage = outlier_analysis.get('outlier_percentage', 0)
        score -= outlier_percentage * 0.3  # 0.3 puntos por cada 1% de outliers
        
        # Penalizar por alta correlación
        correlation_analysis = analysis_results.get('correlation_analysis', {})
        high_correlation_pairs = correlation_analysis.get('correlation_summary', {}).get('high_correlation_pairs', 0)
        score -= high_correlation_pairs * 2  # 2 puntos por cada par altamente correlacionado
        
        return max(0.0, score)
    
    def _extract_key_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extrae insights clave del análisis"""
        insights = []
        
        # Insights sobre valores faltantes
        missing_analysis = analysis_results.get('missing_analysis', {})
        missing_percentage = missing_analysis.get('missing_percentage', 0)
        if missing_percentage > 10:
            insights.append(f"Dataset tiene {missing_percentage:.1f}% de valores faltantes")
        
        # Insights sobre outliers
        outlier_analysis = analysis_results.get('outlier_analysis', {})
        outlier_percentage = outlier_analysis.get('outlier_percentage', 0)
        if outlier_percentage > 5:
            insights.append(f"Dataset tiene {outlier_percentage:.1f}% de outliers")
        
        # Insights sobre correlaciones
        correlation_analysis = analysis_results.get('correlation_analysis', {})
        high_correlation_pairs = correlation_analysis.get('correlation_summary', {}).get('high_correlation_pairs', 0)
        if high_correlation_pairs > 0:
            insights.append(f"Se encontraron {high_correlation_pairs} pares de características altamente correlacionadas")
        
        # Insights sobre dimensionalidad
        dimensionality_analysis = analysis_results.get('dimensionality_analysis', {})
        pca_analysis = dimensionality_analysis.get('pca_analysis', {})
        if pca_analysis:
            n_components = pca_analysis.get('n_components_for_95_variance', 0)
            if n_components < len(analysis_results['data_shape'][1]) * 0.5:
                insights.append(f"Se pueden reducir las características a {n_components} componentes manteniendo 95% de varianza")
        
        return insights
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones sobre valores faltantes
        missing_analysis = analysis_results.get('missing_analysis', {})
        missing_percentage = missing_analysis.get('missing_percentage', 0)
        if missing_percentage > 20:
            recommendations.append("Considerar estrategias de imputación para valores faltantes")
        elif missing_percentage > 5:
            recommendations.append("Revisar estrategias de manejo de valores faltantes")
        
        # Recomendaciones sobre outliers
        outlier_analysis = analysis_results.get('outlier_analysis', {})
        outlier_percentage = outlier_analysis.get('outlier_percentage', 0)
        if outlier_percentage > 10:
            recommendations.append("Investigar y tratar outliers antes del modelado")
        
        # Recomendaciones sobre correlaciones
        correlation_analysis = analysis_results.get('correlation_analysis', {})
        high_correlation_pairs = correlation_analysis.get('correlation_summary', {}).get('high_correlation_pairs', 0)
        if high_correlation_pairs > 5:
            recommendations.append("Considerar eliminar características redundantes")
        
        # Recomendaciones sobre dimensionalidad
        dimensionality_analysis = analysis_results.get('dimensionality_analysis', {})
        pca_analysis = dimensionality_analysis.get('pca_analysis', {})
        if pca_analysis:
            n_components = pca_analysis.get('n_components_for_95_variance', 0)
            if n_components < len(analysis_results['data_shape'][1]) * 0.3:
                recommendations.append("Considerar reducción de dimensionalidad con PCA")
        
        return recommendations
    
    def create_visualizations(self, data: pd.DataFrame, output_dir: str = None, **kwargs) -> Dict[str, str]:
        """
        Crea visualizaciones del dataset
        
        Args:
            data: DataFrame a visualizar
            output_dir: Directorio para guardar gráficos
            **kwargs: Parámetros adicionales
            
        Returns:
            Diccionario con rutas de los gráficos creados
        """
        with self.logger.operation_trace("creacion_visualizaciones"):
            self.logger.info(f"Creando visualizaciones para dataset: {data.shape}")
            
            if output_dir is None:
                output_dir = self.config.data.output_dir
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            plot_paths = {}
            
            try:
                # 1. Distribución de variables numéricas
                plot_paths['numeric_distributions'] = self._plot_numeric_distributions(data, output_dir)
                
                # 2. Matriz de correlación
                plot_paths['correlation_matrix'] = self._plot_correlation_matrix(data, output_dir)
                
                # 3. Análisis de valores faltantes
                plot_paths['missing_values'] = self._plot_missing_values(data, output_dir)
                
                # 4. Análisis de outliers
                plot_paths['outliers'] = self._plot_outliers(data, output_dir)
                
                # 5. Distribución de variables categóricas
                plot_paths['categorical_distributions'] = self._plot_categorical_distributions(data, output_dir)
                
                # 6. Análisis de dimensionalidad
                plot_paths['dimensionality'] = self._plot_dimensionality_analysis(data, output_dir)
                
                self.logger.info(f"Visualizaciones creadas: {len(plot_paths)} gráficos")
                
                return plot_paths
                
            except Exception as e:
                self.logger.log_exception(e, "creacion_visualizaciones")
                raise
    
    def _plot_numeric_distributions(self, data: pd.DataFrame, output_dir: str) -> str:
        """Crea gráficos de distribución para variables numéricas"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return ""
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]
            
            # Histograma
            ax.hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribución de {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
            
            # Agregar estadísticas
            mean_val = data[col].mean()
            median_val = data[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
            ax.legend()
        
        # Ocultar subplots vacíos
        for i in range(len(numeric_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / "numeric_distributions.png"
        plt.savefig(output_path, dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_correlation_matrix(self, data: pd.DataFrame, output_dir: str) -> str:
        """Crea matriz de correlación"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return ""
        
        corr_matrix = data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        
        output_path = Path(output_dir) / "correlation_matrix.png"
        plt.savefig(output_path, dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_missing_values(self, data: pd.DataFrame, output_dir: str) -> str:
        """Crea gráfico de valores faltantes"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Filtrar columnas con valores faltantes
        missing_data = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.values
        })
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_data) == 0:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        ax1.bar(range(len(missing_data)), missing_data['Missing_Percentage'])
        ax1.set_xlabel('Columnas')
        ax1.set_ylabel('Porcentaje de Valores Faltantes')
        ax1.set_title('Porcentaje de Valores Faltantes por Columna')
        ax1.set_xticks(range(len(missing_data)))
        ax1.set_xticklabels(missing_data['Column'], rotation=45, ha='right')
        
        # Gráfico de matriz de valores faltantes
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Matriz de Valores Faltantes')
        plt.tight_layout()
        
        output_path = Path(output_dir) / "missing_values.png"
        plt.savefig(output_path, dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_outliers(self, data: pd.DataFrame, output_dir: str) -> str:
        """Crea gráficos de outliers"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return ""
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]
            
            # Box plot
            ax.boxplot(data[col].dropna())
            ax.set_title(f'Box Plot de {col}')
            ax.set_ylabel(col)
            
            # Agregar puntos de outliers
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[col][(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
            
            if len(outliers) > 0:
                ax.plot(range(1, len(outliers) + 1), outliers, 'ro', alpha=0.5, markersize=3)
        
        # Ocultar subplots vacíos
        for i in range(len(numeric_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / "outliers.png"
        plt.savefig(output_path, dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_categorical_distributions(self, data: pd.DataFrame, output_dir: str) -> str:
        """Crea gráficos de distribución para variables categóricas"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return ""
        
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(categorical_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]
            
            # Gráfico de barras
            value_counts = data[col].value_counts().head(10)  # Top 10 valores
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_title(f'Distribución de {col}')
            ax.set_xlabel('Valores')
            ax.set_ylabel('Frecuencia')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Ocultar subplots vacíos
        for i in range(len(categorical_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / "categorical_distributions.png"
        plt.savefig(output_path, dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_dimensionality_analysis(self, data: pd.DataFrame, output_dir: str) -> str:
        """Crea gráficos de análisis de dimensionalidad"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de varianza explicada por PCA
        try:
            pca = PCA()
            pca.fit(data[numeric_cols])
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            ax1.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, 'bo-')
            ax1.set_xlabel('Número de Componentes')
            ax1.set_ylabel('Varianza Explicada Acumulada')
            ax1.set_title('Análisis PCA - Varianza Explicada')
            ax1.grid(True)
            
            # Marcar punto de 95% de varianza
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Varianza')
            ax1.axvline(x=n_components_95, color='r', linestyle='--', label=f'{n_components_95} componentes')
            ax1.legend()
            
        except Exception as e:
            self.logger.warning(f"Error en gráfico PCA: {e}")
            ax1.text(0.5, 0.5, 'Error en PCA', ha='center', va='center', transform=ax1.transAxes)
        
        # Gráfico de correlación de características
        corr_matrix = data[numeric_cols].corr().abs()
        high_corr_count = (corr_matrix > 0.8).sum() - len(numeric_cols)  # Excluir diagonal
        
        ax2.bar(['Alta Correlación (>0.8)', 'Baja Correlación'], 
                [high_corr_count, len(numeric_cols) * (len(numeric_cols) - 1) - high_corr_count])
        ax2.set_ylabel('Número de Pares')
        ax2.set_title('Análisis de Redundancia de Características')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / "dimensionality_analysis.png"
        plt.savefig(output_path, dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_report(self, analysis_results: Dict[str, Any], output_path: str = None) -> str:
        """
        Genera un reporte completo del análisis exploratorio
        
        Args:
            analysis_results: Resultados del análisis
            output_path: Ruta del archivo de salida
            
        Returns:
            Ruta del reporte generado
        """
        if output_path is None:
            output_path = Path(self.config.data.output_dir) / "exploratory_analysis_report.html"
        
        # Crear reporte HTML
        html_content = self._create_html_report(analysis_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Reporte de análisis exploratorio generado: {output_path}")
        
        return str(output_path)
    
    def _create_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Crea contenido HTML del reporte"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis Exploratorio - PMS 4.0.0</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }
                .insight { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 3px; }
                .recommendation { background-color: #d1ecf1; padding: 10px; margin: 10px 0; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte de Análisis Exploratorio - PMS 4.0.0</h1>
                <p>Fecha: {timestamp}</p>
                <p>Dataset: {shape}</p>
                <p>Score de Calidad: {quality_score:.1f}/100</p>
            </div>
            
            <div class="section">
                <h2>Resumen Ejecutivo</h2>
                <div class="metric">
                    <strong>Score de Calidad:</strong> {quality_score:.1f}/100
                </div>
                <div class="metric">
                    <strong>Valores Faltantes:</strong> {missing_percentage:.1f}%
                </div>
                <div class="metric">
                    <strong>Outliers:</strong> {outlier_percentage:.1f}%
                </div>
            </div>
            
            <div class="section">
                <h2>Insights Clave</h2>
                {insights}
            </div>
            
            <div class="section">
                <h2>Recomendaciones</h2>
                {recommendations}
            </div>
            
            <div class="section">
                <h2>Estadísticas Detalladas</h2>
                <h3>Estadísticas Básicas</h3>
                <table>
                    <tr><th>Métrica</th><th>Valor</th></tr>
                    <tr><td>Filas</td><td>{rows}</td></tr>
                    <tr><td>Columnas</td><td>{columns}</td></tr>
                    <tr><td>Uso de Memoria</td><td>{memory_usage:.2f} MB</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Extraer datos para el template
        summary = analysis_results.get('summary', {})
        basic_stats = analysis_results.get('basic_stats', {})
        missing_analysis = analysis_results.get('missing_analysis', {})
        outlier_analysis = analysis_results.get('outlier_analysis', {})
        
        # Generar insights HTML
        insights_html = ""
        for insight in summary.get('key_insights', []):
            insights_html += f'<div class="insight">💡 {insight}</div>'
        
        # Generar recomendaciones HTML
        recommendations_html = ""
        for rec in summary.get('recommendations', []):
            recommendations_html += f'<div class="recommendation">📋 {rec}</div>'
        
        # Llenar template
        html_content = html_template.format(
            timestamp=analysis_results.get('timestamp', 'N/A'),
            shape=f"{analysis_results.get('data_shape', (0, 0))[0]} filas × {analysis_results.get('data_shape', (0, 0))[1]} columnas",
            quality_score=summary.get('data_quality_score', 0),
            missing_percentage=missing_analysis.get('missing_percentage', 0),
            outlier_percentage=outlier_analysis.get('outlier_percentage', 0),
            insights=insights_html,
            recommendations=recommendations_html,
            rows=basic_stats.get('shape', (0, 0))[0],
            columns=basic_stats.get('shape', (0, 0))[1],
            memory_usage=basic_stats.get('memory_usage_mb', 0)
        )
        
        return html_content
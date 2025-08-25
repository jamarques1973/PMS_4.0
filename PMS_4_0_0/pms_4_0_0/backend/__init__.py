"""
Capa Backend - Lógica de Negocio PMS 4.0.0
==========================================

Esta capa contiene toda la lógica de negocio:
- Procesamiento de datos
- Entrenamiento de modelos
- Optimización de hiperparámetros
- Análisis de interpretabilidad
- Generación de informes
"""

from .data import DataProcessor
from .models import ModelManager
from .optimization import OptimizationEngine
from .xai import XAIAnalyzer
from .reporting import ReportGenerator

__all__ = [
    "DataProcessor",
    "ModelManager", 
    "OptimizationEngine",
    "XAIAnalyzer",
    "ReportGenerator"
]
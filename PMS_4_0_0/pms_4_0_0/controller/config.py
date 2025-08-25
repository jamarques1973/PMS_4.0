"""
Sistema de Configuración Centralizado - PMS 4.0.0
================================================

Este módulo maneja toda la configuración del sistema de manera centralizada,
permitiendo máxima flexibilidad y personalización de parámetros.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SystemConfig:
    """Configuración del sistema principal"""
    name: str = "PMS 4.0.0"
    version: str = "4.0.0"
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    temp_dir: str = "./temp"
    output_dir: str = "./output"


@dataclass
class ModelConfig:
    """Configuración de modelos"""
    svr: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "default_params": {
            "C": 1.0,
            "epsilon": 0.1,
            "kernel": "rbf",
            "gamma": "scale"
        }
    })
    
    neural_network: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "default_params": {
            "layers": [64, 32],
            "activation": "relu",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "dropout": 0.2
        }
    })
    
    xgboost: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0
        }
    })
    
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "default_params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
    })
    
    rnn: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "default_params": {
            "units": 50,
            "layers": 1,
            "window_size": 10,
            "batch_size": 32,
            "epochs": 30
        }
    })


@dataclass
class OptimizationConfig:
    """Configuración de optimización"""
    enabled: bool = True
    max_trials: int = 50
    timeout: int = 3600  # segundos
    engines: Dict[str, Any] = field(default_factory=lambda: {
        "grid_search": {"enabled": True},
        "random_search": {"enabled": True, "n_iter": 30},
        "bayesian": {"enabled": True, "n_iter": 30},
        "optuna": {"enabled": True, "n_trials": 50},
        "hyperband": {"enabled": True}
    })


@dataclass
class XAIConfig:
    """Configuración de XAI"""
    enabled: bool = True
    methods: Dict[str, Any] = field(default_factory=lambda: {
        "shap": {"enabled": True, "n_samples": 100},
        "lime": {"enabled": True, "n_samples": 100},
        "permutation": {"enabled": True, "n_repeats": 10},
        "pdp": {"enabled": True, "grid_resolution": 20},
        "ale": {"enabled": True, "n_bins": 20},
        "ice": {"enabled": True, "n_samples": 50},
        "counterfactual": {"enabled": True, "n_samples": 50},
        "anchors": {"enabled": True, "n_samples": 50},
        "surrogate": {"enabled": True, "n_samples": 100},
        "ebm": {"enabled": True, "max_rounds": 100}
    })


@dataclass
class UIConfig:
    """Configuración de la interfaz de usuario"""
    theme: str = "default"
    language: str = "es"
    layout: str = "standard"
    widgets: Dict[str, Any] = field(default_factory=lambda: {
        "data_loader": {"enabled": True},
        "model_trainer": {"enabled": True},
        "optimizer": {"enabled": True},
        "xai_analyzer": {"enabled": True},
        "report_generator": {"enabled": True}
    })


class Config:
    """
    Clase principal de configuración del sistema PMS 4.0.0
    
    Esta clase centraliza toda la configuración del sistema, permitiendo:
    - Carga desde archivos YAML/JSON
    - Configuración por defecto
    - Validación de parámetros
    - Sobrescritura dinámica
    - Persistencia de configuración
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la configuración del sistema
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_path = config_path
        self._config_data = {}
        
        # Configuraciones por defecto
        self.system = SystemConfig()
        self.models = ModelConfig()
        self.optimization = OptimizationConfig()
        self.xai = XAIConfig()
        self.ui = UIConfig()
        
        # Cargar configuración
        self._load_config()
        
        # Configurar logging
        self._setup_logging()
        
        logger.info(f"Configuración PMS 4.0.0 inicializada: {self.system.name} v{self.system.version}")
    
    def _load_config(self):
        """Carga la configuración desde archivo o usa valores por defecto"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        self._config_data = yaml.safe_load(f)
                    elif self.config_path.endswith('.json'):
                        self._config_data = json.load(f)
                    else:
                        raise ValueError(f"Formato de archivo no soportado: {self.config_path}")
                
                # Aplicar configuración cargada
                self._apply_config()
                logger.info(f"Configuración cargada desde: {self.config_path}")
                
            except Exception as e:
                logger.warning(f"No se pudo cargar la configuración desde {self.config_path}: {e}")
                logger.info("Usando configuración por defecto")
        else:
            logger.info("Usando configuración por defecto")
    
    def _apply_config(self):
        """Aplica la configuración cargada a los objetos de configuración"""
        if 'system' in self._config_data:
            for key, value in self._config_data['system'].items():
                if hasattr(self.system, key):
                    setattr(self.system, key, value)
        
        if 'models' in self._config_data:
            for model_name, model_config in self._config_data['models'].items():
                if hasattr(self.models, model_name):
                    model_obj = getattr(self.models, model_name)
                    if isinstance(model_obj, dict):
                        model_obj.update(model_config)
        
        if 'optimization' in self._config_data:
            for key, value in self._config_data['optimization'].items():
                if hasattr(self.optimization, key):
                    setattr(self.optimization, key, value)
        
        if 'xai' in self._config_data:
            for key, value in self._config_data['xai'].items():
                if hasattr(self.xai, key):
                    setattr(self.xai, key, value)
        
        if 'ui' in self._config_data:
            for key, value in self._config_data['ui'].items():
                if hasattr(self.ui, key):
                    setattr(self.ui, key, value)
    
    def _setup_logging(self):
        """Configura el sistema de logging basado en la configuración"""
        log_level = self.system.log_level.upper()
        
        # Configurar loguru
        logger.remove()  # Remover handlers por defecto
        logger.add(
            "logs/pms_{time}.log",
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        
        if self.system.debug:
            logger.add(
                lambda msg: print(msg, end=""),
                level=log_level,
                format="{time:HH:mm:ss} | {level} | {message}"
            )
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Obtiene la configuración de un modelo específico
        
        Args:
            model_name: Nombre del modelo (svr, neural_network, xgboost, etc.)
            
        Returns:
            Configuración del modelo
        """
        if hasattr(self.models, model_name):
            return getattr(self.models, model_name)
        else:
            logger.warning(f"Modelo no encontrado: {model_name}")
            return {}
    
    def is_model_enabled(self, model_name: str) -> bool:
        """
        Verifica si un modelo está habilitado
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            True si el modelo está habilitado
        """
        config = self.get_model_config(model_name)
        return config.get('enabled', False)
    
    def get_xai_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        Obtiene la configuración de un método XAI específico
        
        Args:
            method_name: Nombre del método XAI
            
        Returns:
            Configuración del método
        """
        return self.xai.methods.get(method_name, {})
    
    def is_xai_method_enabled(self, method_name: str) -> bool:
        """
        Verifica si un método XAI está habilitado
        
        Args:
            method_name: Nombre del método
            
        Returns:
            True si el método está habilitado
        """
        config = self.get_xai_method_config(method_name)
        return config.get('enabled', False)
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Actualiza dinámicamente una configuración
        
        Args:
            section: Sección de configuración (system, models, etc.)
            key: Clave a actualizar
            value: Nuevo valor
        """
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                logger.info(f"Configuración actualizada: {section}.{key} = {value}")
            else:
                logger.warning(f"Clave no encontrada: {section}.{key}")
        else:
            logger.warning(f"Sección no encontrada: {section}")
    
    def save_config(self, output_path: str):
        """
        Guarda la configuración actual en un archivo
        
        Args:
            output_path: Ruta donde guardar la configuración
        """
        config_dict = {
            'system': self.system.__dict__,
            'models': {
                'svr': self.models.svr,
                'neural_network': self.models.neural_network,
                'xgboost': self.models.xgboost,
                'random_forest': self.models.random_forest,
                'rnn': self.models.rnn
            },
            'optimization': self.optimization.__dict__,
            'xai': self.xai.__dict__,
            'ui': self.ui.__dict__
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif output_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Formato de archivo no soportado: {output_path}")
            
            logger.info(f"Configuración guardada en: {output_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar la configuración: {e}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Obtiene toda la configuración como diccionario
        
        Returns:
            Diccionario con toda la configuración
        """
        return {
            'system': self.system.__dict__,
            'models': {
                'svr': self.models.svr,
                'neural_network': self.models.neural_network,
                'xgboost': self.models.xgboost,
                'random_forest': self.models.random_forest,
                'rnn': self.models.rnn
            },
            'optimization': self.optimization.__dict__,
            'xai': self.xai.__dict__,
            'ui': self.ui.__dict__
        }
    
    def validate_config(self) -> bool:
        """
        Valida la configuración actual
        
        Returns:
            True si la configuración es válida
        """
        try:
            # Validar directorios
            for dir_path in [self.system.cache_dir, self.system.temp_dir, self.system.output_dir]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Validar modelos habilitados
            enabled_models = []
            for model_name in ['svr', 'neural_network', 'xgboost', 'random_forest', 'rnn']:
                if self.is_model_enabled(model_name):
                    enabled_models.append(model_name)
            
            if not enabled_models:
                logger.warning("No hay modelos habilitados")
            
            # Validar métodos XAI habilitados
            enabled_xai = []
            for method_name in self.xai.methods.keys():
                if self.is_xai_method_enabled(method_name):
                    enabled_xai.append(method_name)
            
            if not enabled_xai:
                logger.warning("No hay métodos XAI habilitados")
            
            logger.info(f"Configuración válida. Modelos habilitados: {enabled_models}")
            logger.info(f"Métodos XAI habilitados: {enabled_xai}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en la validación de configuración: {e}")
            return False
"""
Orquestador Principal - PMS 4.0.0
================================

Este módulo actúa como el cerebro del sistema, coordinando todas las capas:
- Frontend (UI)
- Backend (Lógica de Negocio)
- Controlador (Configuración y Logging)

Proporciona una interfaz unificada para ejecutar pipelines completos
y gestionar flujos de trabajo complejos.
"""

import os
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .logger import Logger, get_logger
from ..backend.data import DataProcessor
from ..backend.models import ModelManager
from ..backend.optimization import OptimizationEngine
from ..backend.xai import XAIAnalyzer
from ..backend.reporting import ReportGenerator
from ..frontend.widgets import WidgetManager
from ..frontend.themes import ThemeManager
from ..frontend.layouts import LayoutManager


@dataclass
class PipelineStep:
    """Representa un paso en el pipeline"""
    name: str
    function: Callable
    dependencies: List[str] = None
    enabled: bool = True
    timeout: int = 300  # segundos
    retries: int = 3


class Orchestrator:
    """
    Orquestador principal del sistema PMS 4.0.0
    
    Responsabilidades:
    - Coordinación entre capas Frontend y Backend
    - Gestión de flujos de trabajo
    - Manejo de errores y logging
    - Configuración del sistema
    - Ejecución de pipelines
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el orquestador
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        # Configuración y logging
        self.config = Config(config_path)
        self.logger = Logger(
            log_level=self.config.system.log_level,
            log_dir=self.config.system.output_dir
        )
        
        # Validar configuración
        if not self.config.validate_config():
            raise ValueError("Configuración inválida")
        
        # Registrar configuración
        self.logger.log_configuration(self.config.get_all_config())
        
        # Inicializar componentes del backend
        self._init_backend_components()
        
        # Inicializar componentes del frontend
        self._init_frontend_components()
        
        # Pipeline steps
        self.pipeline_steps = self._create_pipeline_steps()
        
        # Estado del sistema
        self.is_running = False
        self.current_pipeline = None
        self.execution_history = []
        
        # Thread pool para operaciones paralelas
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.system.max_workers
        )
        
        self.logger.info("Orquestador PMS 4.0.0 inicializado correctamente")
    
    def _init_backend_components(self):
        """Inicializa los componentes del backend"""
        with self.logger.operation_trace("inicializar_backend"):
            self.data_processor = DataProcessor(self.config, self.logger)
            self.model_manager = ModelManager(self.config, self.logger)
            self.optimization_engine = OptimizationEngine(self.config, self.logger)
            self.xai_analyzer = XAIAnalyzer(self.config, self.logger)
            self.report_generator = ReportGenerator(self.config, self.logger)
            
            self.logger.info("Componentes del backend inicializados")
    
    def _init_frontend_components(self):
        """Inicializa los componentes del frontend"""
        with self.logger.operation_trace("inicializar_frontend"):
            self.widget_manager = WidgetManager(self.config, self.logger)
            self.theme_manager = ThemeManager(self.config, self.logger)
            self.layout_manager = LayoutManager(self.config, self.logger)
            
            self.logger.info("Componentes del frontend inicializados")
    
    def _create_pipeline_steps(self) -> Dict[str, PipelineStep]:
        """Crea los pasos del pipeline"""
        return {
            "data_loading": PipelineStep(
                name="Carga de Datos",
                function=self.data_processor.load_data,
                dependencies=[],
                enabled=True
            ),
            "data_preprocessing": PipelineStep(
                name="Preprocesamiento de Datos",
                function=self.data_processor.preprocess_data,
                dependencies=["data_loading"],
                enabled=True
            ),
            "feature_selection": PipelineStep(
                name="Selección de Características",
                function=self.data_processor.select_features,
                dependencies=["data_preprocessing"],
                enabled=True
            ),
            "model_training": PipelineStep(
                name="Entrenamiento de Modelos",
                function=self.model_manager.train_all_models,
                dependencies=["feature_selection"],
                enabled=True
            ),
            "model_evaluation": PipelineStep(
                name="Evaluación de Modelos",
                function=self.model_manager.evaluate_all_models,
                dependencies=["model_training"],
                enabled=True
            ),
            "optimization": PipelineStep(
                name="Optimización de Hiperparámetros",
                function=self.optimization_engine.optimize_all_models,
                dependencies=["model_evaluation"],
                enabled=self.config.optimization.enabled
            ),
            "xai_analysis": PipelineStep(
                name="Análisis XAI",
                function=self.xai_analyzer.analyze_all_models,
                dependencies=["model_training"],
                enabled=self.config.xai.enabled
            ),
            "report_generation": PipelineStep(
                name="Generación de Informes",
                function=self.report_generator.generate_comprehensive_report,
                dependencies=["model_evaluation", "xai_analysis"],
                enabled=True
            )
        }
    
    def run_pipeline(self, pipeline_name: str = "complete", 
                    steps: Optional[List[str]] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un pipeline completo
        
        Args:
            pipeline_name: Nombre del pipeline
            steps: Lista de pasos específicos a ejecutar
            **kwargs: Parámetros adicionales
            
        Returns:
            Resultados del pipeline
        """
        if self.is_running:
            raise RuntimeError("Ya hay un pipeline ejecutándose")
        
        self.is_running = True
        self.current_pipeline = pipeline_name
        
        try:
            with self.logger.operation_trace(f"pipeline_{pipeline_name}"):
                self.logger.info(f"Iniciando pipeline: {pipeline_name}")
                
                # Determinar pasos a ejecutar
                if steps is None:
                    steps_to_execute = list(self.pipeline_steps.keys())
                else:
                    steps_to_execute = [s for s in steps if s in self.pipeline_steps]
                
                # Validar dependencias
                self._validate_pipeline_dependencies(steps_to_execute)
                
                # Ejecutar pasos
                results = {}
                execution_order = self._get_execution_order(steps_to_execute)
                
                for step_name in execution_order:
                    if step_name in steps_to_execute:
                        step_result = self._execute_pipeline_step(step_name, **kwargs)
                        results[step_name] = step_result
                
                # Registrar ejecución
                execution_record = {
                    'pipeline_name': pipeline_name,
                    'steps_executed': steps_to_execute,
                    'results': results,
                    'timestamp': time.time(),
                    'success': True
                }
                self.execution_history.append(execution_record)
                
                self.logger.info(f"Pipeline '{pipeline_name}' completado exitosamente")
                return results
                
        except Exception as e:
            self.logger.log_exception(e, f"pipeline_{pipeline_name}")
            execution_record = {
                'pipeline_name': pipeline_name,
                'steps_executed': steps_to_execute if 'steps_to_execute' in locals() else [],
                'error': str(e),
                'timestamp': time.time(),
                'success': False
            }
            self.execution_history.append(execution_record)
            raise
        
        finally:
            self.is_running = False
            self.current_pipeline = None
    
    def _validate_pipeline_dependencies(self, steps: List[str]):
        """Valida las dependencias del pipeline"""
        for step_name in steps:
            step = self.pipeline_steps[step_name]
            if step.dependencies:
                for dep in step.dependencies:
                    if dep not in steps:
                        raise ValueError(f"Paso '{step_name}' requiere '{dep}' pero no está incluido")
    
    def _get_execution_order(self, steps: List[str]) -> List[str]:
        """Obtiene el orden de ejecución basado en dependencias"""
        # Implementación simple de ordenamiento topológico
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(step_name):
            if step_name in temp_visited:
                raise ValueError(f"Dependencia circular detectada: {step_name}")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            step = self.pipeline_steps[step_name]
            
            if step.dependencies:
                for dep in step.dependencies:
                    if dep in self.pipeline_steps:
                        dfs(dep)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in steps:
            if step_name not in visited:
                dfs(step_name)
        
        return order
    
    def _execute_pipeline_step(self, step_name: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta un paso específico del pipeline"""
        step = self.pipeline_steps[step_name]
        
        if not step.enabled:
            self.logger.info(f"Paso '{step_name}' está deshabilitado, saltando...")
            return {'status': 'skipped', 'reason': 'disabled'}
        
        with self.logger.operation_trace(f"pipeline_step_{step_name}"):
            self.logger.info(f"Ejecutando paso: {step.name}")
            
            try:
                # Ejecutar función con timeout
                future = self.executor.submit(step.function, **kwargs)
                result = future.result(timeout=step.timeout)
                
                self.logger.info(f"Paso '{step.name}' completado exitosamente")
                return {
                    'status': 'success',
                    'result': result,
                    'step_name': step.name
                }
                
            except Exception as e:
                self.logger.log_exception(e, f"pipeline_step_{step_name}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'step_name': step.name
                }
    
    def run_single_step(self, step_name: str, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un paso individual del pipeline
        
        Args:
            step_name: Nombre del paso
            **kwargs: Parámetros del paso
            
        Returns:
            Resultado del paso
        """
        if step_name not in self.pipeline_steps:
            raise ValueError(f"Paso no encontrado: {step_name}")
        
        return self._execute_pipeline_step(step_name, **kwargs)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del pipeline
        
        Returns:
            Estado del pipeline
        """
        return {
            'is_running': self.is_running,
            'current_pipeline': self.current_pipeline,
            'available_steps': list(self.pipeline_steps.keys()),
            'enabled_steps': [name for name, step in self.pipeline_steps.items() if step.enabled],
            'execution_history_count': len(self.execution_history),
            'last_execution': self.execution_history[-1] if self.execution_history else None
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de ejecuciones
        
        Args:
            limit: Número máximo de registros a retornar
            
        Returns:
            Historial de ejecuciones
        """
        return self.execution_history[-limit:] if self.execution_history else []
    
    def run_parallel_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ejecuta operaciones en paralelo
        
        Args:
            operations: Lista de operaciones a ejecutar
            
        Returns:
            Resultados de las operaciones
        """
        with self.logger.operation_trace("operaciones_paralelas"):
            futures = []
            results = []
            
            # Enviar operaciones al thread pool
            for op in operations:
                if 'function' in op and 'args' in op:
                    future = self.executor.submit(op['function'], *op['args'], **op.get('kwargs', {}))
                    futures.append((future, op.get('name', 'unknown')))
            
            # Recopilar resultados
            for future, name in futures:
                try:
                    result = future.result(timeout=300)  # 5 minutos por operación
                    results.append({
                        'name': name,
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    self.logger.log_exception(e, f"operacion_paralela_{name}")
                    results.append({
                        'name': name,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return results
    
    def create_custom_pipeline(self, name: str, steps: List[str], 
                             description: str = "") -> str:
        """
        Crea un pipeline personalizado
        
        Args:
            name: Nombre del pipeline
            steps: Lista de pasos
            description: Descripción del pipeline
            
        Returns:
            ID del pipeline creado
        """
        # Validar pasos
        for step in steps:
            if step not in self.pipeline_steps:
                raise ValueError(f"Paso no válido: {step}")
        
        # Crear pipeline personalizado
        custom_pipeline = {
            'name': name,
            'description': description,
            'steps': steps,
            'created_at': time.time()
        }
        
        # Guardar pipeline personalizado
        pipeline_file = Path(self.config.system.output_dir) / f"pipeline_{name}.json"
        import json
        with open(pipeline_file, 'w', encoding='utf-8') as f:
            json.dump(custom_pipeline, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline personalizado creado: {name}")
        return name
    
    def load_custom_pipeline(self, name: str) -> Dict[str, Any]:
        """
        Carga un pipeline personalizado
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Configuración del pipeline
        """
        pipeline_file = Path(self.config.system.output_dir) / f"pipeline_{name}.json"
        
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Pipeline no encontrado: {name}")
        
        import json
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_config = json.load(f)
        
        self.logger.info(f"Pipeline personalizado cargado: {name}")
        return pipeline_config
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Obtiene el estado de salud del sistema
        
        Returns:
            Estado de salud del sistema
        """
        health_info = {
            'system': {
                'status': 'healthy',
                'uptime': time.time() - getattr(self, '_start_time', time.time()),
                'memory_usage': self.logger.get_system_info(),
                'config_valid': self.config.validate_config()
            },
            'components': {
                'data_processor': hasattr(self, 'data_processor'),
                'model_manager': hasattr(self, 'model_manager'),
                'optimization_engine': hasattr(self, 'optimization_engine'),
                'xai_analyzer': hasattr(self, 'xai_analyzer'),
                'report_generator': hasattr(self, 'report_generator'),
                'widget_manager': hasattr(self, 'widget_manager'),
                'theme_manager': hasattr(self, 'theme_manager'),
                'layout_manager': hasattr(self, 'layout_manager')
            },
            'pipeline': {
                'is_running': self.is_running,
                'current_pipeline': self.current_pipeline,
                'total_steps': len(self.pipeline_steps),
                'enabled_steps': len([s for s in self.pipeline_steps.values() if s.enabled])
            },
            'performance': self.logger.performance_monitor.get_performance_summary()
        }
        
        return health_info
    
    def shutdown(self):
        """Apaga el orquestador de manera segura"""
        self.logger.info("Apagando orquestador PMS 4.0.0...")
        
        # Detener pipeline si está ejecutándose
        if self.is_running:
            self.logger.warning("Pipeline en ejecución será interrumpido")
            self.is_running = False
        
        # Cerrar thread pool
        self.executor.shutdown(wait=True)
        
        # Exportar logs finales
        self.logger.export_logs(
            Path(self.config.system.output_dir) / "final_logs.json"
        )
        
        self.logger.info("Orquestador PMS 4.0.0 apagado correctamente")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
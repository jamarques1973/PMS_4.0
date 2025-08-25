"""
Punto de Entrada Principal - PMS 4.0.0
=====================================

Este módulo proporciona la interfaz principal del sistema PMS 4.0.0,
actuando como punto de entrada para todas las funcionalidades.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse

from .controller.orchestrator import Orchestrator
from .controller.config import Config
from .controller.logger import Logger, get_logger


class PMSSystem:
    """
    Clase principal del sistema PMS 4.0.0
    
    Esta clase proporciona una interfaz unificada para acceder a todas
    las funcionalidades del sistema de manera modular y organizada.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el sistema PMS 4.0.0
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        # Inicializar orquestador
        self.orchestrator = Orchestrator(config_path)
        
        # Referencias directas a componentes para acceso rápido
        self.config = self.orchestrator.config
        self.logger = self.orchestrator.logger
        
        # Componentes del backend
        self.data = self.orchestrator.data_processor
        self.models = self.orchestrator.model_manager
        self.optimization = self.orchestrator.optimization_engine
        self.xai = self.orchestrator.xai_analyzer
        self.reporting = self.orchestrator.report_generator
        
        # Componentes del frontend
        self.widgets = self.orchestrator.widget_manager
        self.themes = self.orchestrator.theme_manager
        self.layouts = self.orchestrator.layout_manager
        
        self.logger.info("Sistema PMS 4.0.0 inicializado correctamente")
    
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
        return self.orchestrator.run_pipeline(pipeline_name, steps, **kwargs)
    
    def run_single_step(self, step_name: str, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un paso individual del pipeline
        
        Args:
            step_name: Nombre del paso
            **kwargs: Parámetros del paso
            
        Returns:
            Resultado del paso
        """
        return self.orchestrator.run_single_step(step_name, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del sistema
        
        Returns:
            Estado del sistema
        """
        return {
            'system_info': {
                'name': self.config.system.name,
                'version': self.config.system.version,
                'status': 'running'
            },
            'pipeline_status': self.orchestrator.get_pipeline_status(),
            'system_health': self.orchestrator.get_system_health(),
            'configuration': {
                'models_enabled': [name for name in ['svr', 'neural_network', 'xgboost', 'random_forest', 'rnn'] 
                                 if self.config.is_model_enabled(name)],
                'xai_methods_enabled': [name for name in self.config.xai.methods.keys() 
                                       if self.config.is_xai_method_enabled(name)],
                'optimization_enabled': self.config.optimization.enabled
            }
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de ejecuciones
        
        Args:
            limit: Número máximo de registros a retornar
            
        Returns:
            Historial de ejecuciones
        """
        return self.orchestrator.get_execution_history(limit)
    
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
        return self.orchestrator.create_custom_pipeline(name, steps, description)
    
    def load_custom_pipeline(self, name: str) -> Dict[str, Any]:
        """
        Carga un pipeline personalizado
        
        Args:
            name: Nombre del pipeline
            
        Returns:
            Configuración del pipeline
        """
        return self.orchestrator.load_custom_pipeline(name)
    
    def export_logs(self, output_path: str, format: str = "json"):
        """
        Exporta logs del sistema
        
        Args:
            output_path: Ruta del archivo de salida
            format: Formato de exportación
        """
        self.logger.export_logs(output_path, format)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de logs
        
        Returns:
            Resumen de logs
        """
        return self.logger.get_log_summary()
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Actualiza la configuración dinámicamente
        
        Args:
            section: Sección de configuración
            key: Clave a actualizar
            value: Nuevo valor
        """
        self.config.update_config(section, key, value)
    
    def save_config(self, output_path: str):
        """
        Guarda la configuración actual
        
        Args:
            output_path: Ruta donde guardar la configuración
        """
        self.config.save_config(output_path)
    
    def shutdown(self):
        """Apaga el sistema de manera segura"""
        self.logger.info("Apagando sistema PMS 4.0.0...")
        self.orchestrator.shutdown()
        self.logger.info("Sistema PMS 4.0.0 apagado correctamente")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


def create_default_config(output_path: str = "config.yaml"):
    """
    Crea un archivo de configuración por defecto
    
    Args:
        output_path: Ruta donde guardar la configuración
    """
    config = Config()
    config.save_config(output_path)
    print(f"Configuración por defecto creada en: {output_path}")


def validate_installation():
    """
    Valida la instalación del sistema
    
    Returns:
        True si la instalación es válida
    """
    try:
        # Verificar dependencias básicas
        import pandas
        import numpy
        import sklearn
        import tensorflow
        import xgboost
        import ipywidgets
        import loguru
        
        print("✅ Todas las dependencias básicas están instaladas")
        
        # Verificar estructura de directorios
        required_dirs = ['logs', 'cache', 'temp', 'output']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        print("✅ Estructura de directorios creada")
        
        # Verificar configuración
        config = Config()
        if config.validate_config():
            print("✅ Configuración válida")
        else:
            print("⚠️ Configuración tiene problemas")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return False


def main():
    """
    Función principal para ejecución desde línea de comandos
    """
    parser = argparse.ArgumentParser(
        description="PMS 4.0.0 - Pipeline Modeling Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  pms --config config.yaml --pipeline complete
  pms --validate
  pms --create-config my_config.yaml
  pms --step data_loading
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Ruta al archivo de configuración'
    )
    
    parser.add_argument(
        '--pipeline', '-p',
        type=str,
        default='complete',
        help='Nombre del pipeline a ejecutar'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        help='Pasos específicos del pipeline a ejecutar'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validar la instalación del sistema'
    )
    
    parser.add_argument(
        '--create-config',
        type=str,
        help='Crear archivo de configuración por defecto'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        help='Ejecutar un paso específico del pipeline'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Mostrar estado del sistema'
    )
    
    parser.add_argument(
        '--export-logs',
        type=str,
        help='Exportar logs a archivo'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Nivel de logging'
    )
    
    args = parser.parse_args()
    
    # Validar instalación
    if args.validate:
        if validate_installation():
            print("✅ Instalación válida")
            sys.exit(0)
        else:
            print("❌ Instalación inválida")
            sys.exit(1)
    
    # Crear configuración por defecto
    if args.create_config:
        create_default_config(args.create_config)
        sys.exit(0)
    
    # Configurar logging
    logger = Logger(log_level=args.log_level)
    
    try:
        # Inicializar sistema
        pms = PMSSystem(args.config)
        
        # Mostrar estado
        if args.status:
            status = pms.get_status()
            import json
            print(json.dumps(status, indent=2, ensure_ascii=False))
            sys.exit(0)
        
        # Exportar logs
        if args.export_logs:
            pms.export_logs(args.export_logs)
            sys.exit(0)
        
        # Ejecutar paso específico
        if args.step:
            result = pms.run_single_step(args.step)
            print(f"Resultado del paso '{args.step}': {result}")
            sys.exit(0)
        
        # Ejecutar pipeline
        if args.pipeline:
            result = pms.run_pipeline(args.pipeline, args.steps)
            print(f"Pipeline '{args.pipeline}' completado exitosamente")
            print(f"Resultados: {result}")
            sys.exit(0)
        
        # Si no se especificó ninguna acción, mostrar ayuda
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\n⚠️ Operación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
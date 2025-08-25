"""
Sistema de Logging Avanzado - PMS 4.0.0
======================================

Este módulo proporciona un sistema de logging completo con:
- Trazas detalladas de ejecución
- Logs estructurados
- Debug avanzado
- Monitorización en tiempo real
- Exportación de logs
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from functools import wraps
import threading
from loguru import logger
import psutil


class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self):
        self.start_time = None
        self.memory_start = None
        self.cpu_start = None
        self.measurements = []
    
    def start_monitoring(self):
        """Inicia el monitoreo de rendimiento"""
        self.start_time = time.time()
        self.memory_start = psutil.virtual_memory().used
        self.cpu_start = psutil.cpu_percent()
        logger.debug("Monitoreo de rendimiento iniciado")
    
    def end_monitoring(self, operation_name: str):
        """Finaliza el monitoreo y registra métricas"""
        if self.start_time is None:
            return
        
        end_time = time.time()
        memory_end = psutil.virtual_memory().used
        cpu_end = psutil.cpu_percent()
        
        duration = end_time - self.start_time
        memory_used = memory_end - self.memory_start
        cpu_avg = (self.cpu_start + cpu_end) / 2
        
        measurement = {
            'operation': operation_name,
            'duration': duration,
            'memory_used_mb': memory_used / (1024 * 1024),
            'cpu_percent': cpu_avg,
            'timestamp': datetime.now().isoformat()
        }
        
        self.measurements.append(measurement)
        
        logger.info(f"Operación '{operation_name}' completada en {duration:.2f}s "
                   f"(Memoria: {memory_used/(1024*1024):.1f}MB, CPU: {cpu_avg:.1f}%)")
        
        # Reset para próxima operación
        self.start_time = None
        self.memory_start = None
        self.cpu_start = None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de rendimiento"""
        if not self.measurements:
            return {}
        
        total_duration = sum(m['duration'] for m in self.measurements)
        total_memory = sum(m['memory_used_mb'] for m in self.measurements)
        avg_cpu = sum(m['cpu_percent'] for m in self.measurements) / len(self.measurements)
        
        return {
            'total_operations': len(self.measurements),
            'total_duration': total_duration,
            'total_memory_mb': total_memory,
            'average_cpu_percent': avg_cpu,
            'operations': self.measurements
        }


class Logger:
    """
    Sistema de logging avanzado para PMS 4.0.0
    
    Características:
    - Logging estructurado
    - Trazas de ejecución
    - Monitorización de rendimiento
    - Debug avanzado
    - Exportación de logs
    - Logs en tiempo real
    """
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        """
        Inicializa el sistema de logging
        
        Args:
            log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directorio para almacenar logs
        """
        self.log_level = log_level.upper()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitor de rendimiento
        self.performance_monitor = PerformanceMonitor()
        
        # Configurar loguru
        self._setup_loguru()
        
        # Contadores de eventos
        self.event_counters = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'debug': 0,
            'critical': 0
        }
        
        # Logs en memoria para exportación
        self.memory_logs = []
        self.max_memory_logs = 1000
        
        logger.info(f"Sistema de logging PMS 4.0.0 inicializado (Nivel: {self.log_level})")
    
    def _setup_loguru(self):
        """Configura loguru con múltiples handlers"""
        # Remover handlers por defecto
        logger.remove()
        
        # Handler para archivo principal
        logger.add(
            self.log_dir / "pms_{time:YYYY-MM-DD}.log",
            level=self.log_level,
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            backtrace=True,
            diagnose=True
        )
        
        # Handler para errores
        logger.add(
            self.log_dir / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            backtrace=True,
            diagnose=True
        )
        
        # Handler para debug
        if self.log_level == "DEBUG":
            logger.add(
                self.log_dir / "debug_{time:YYYY-MM-DD}.log",
                level="DEBUG",
                rotation="5 MB",
                retention="7 days",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                backtrace=True,
                diagnose=True
            )
        
        # Handler para consola
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True
        )
    
    def log_event(self, level: str, message: str, **kwargs):
        """
        Registra un evento con metadatos adicionales
        
        Args:
            level: Nivel del log
            message: Mensaje principal
            **kwargs: Metadatos adicionales
        """
        # Incrementar contador
        if level.lower() in self.event_counters:
            self.event_counters[level.lower()] += 1
        
        # Crear log estructurado
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.upper(),
            'message': message,
            'thread': threading.current_thread().name,
            'process': os.getpid(),
            **kwargs
        }
        
        # Agregar a logs en memoria
        self.memory_logs.append(log_entry)
        if len(self.memory_logs) > self.max_memory_logs:
            self.memory_logs.pop(0)
        
        # Log con loguru
        if level.upper() == "DEBUG":
            logger.debug(message)
        elif level.upper() == "INFO":
            logger.info(message)
        elif level.upper() == "WARNING":
            logger.warning(message)
        elif level.upper() == "ERROR":
            logger.error(message)
        elif level.upper() == "CRITICAL":
            logger.critical(message)
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self.log_event("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log de información"""
        self.log_event("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de advertencia"""
        self.log_event("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log de error"""
        self.log_event("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log crítico"""
        self.log_event("CRITICAL", message, **kwargs)
    
    @contextmanager
    def operation_trace(self, operation_name: str, **kwargs):
        """
        Context manager para trazar operaciones
        
        Args:
            operation_name: Nombre de la operación
            **kwargs: Metadatos adicionales
        """
        start_time = time.time()
        self.info(f"Iniciando operación: {operation_name}", operation=operation_name, **kwargs)
        
        try:
            self.performance_monitor.start_monitoring()
            yield
            self.performance_monitor.end_monitoring(operation_name)
            self.info(f"Operación completada exitosamente: {operation_name}", 
                     operation=operation_name, duration=time.time() - start_time, **kwargs)
        except Exception as e:
            duration = time.time() - start_time
            self.error(f"Error en operación: {operation_name}", 
                      operation=operation_name, error=str(e), duration=duration, **kwargs)
            self.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
    def trace_function(self, func):
        """
        Decorador para trazar funciones
        
        Args:
            func: Función a decorar
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module_name = func.__module__
            
            with self.operation_trace(f"{module_name}.{func_name}"):
                return func(*args, **kwargs)
        
        return wrapper
    
    def log_exception(self, exception: Exception, context: str = "", **kwargs):
        """
        Registra una excepción con contexto
        
        Args:
            exception: Excepción capturada
            context: Contexto donde ocurrió
            **kwargs: Metadatos adicionales
        """
        self.error(f"Excepción en {context}: {str(exception)}", 
                  exception_type=type(exception).__name__,
                  exception_message=str(exception),
                  context=context,
                  traceback=traceback.format_exc(),
                  **kwargs)
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Registra métricas de rendimiento
        
        Args:
            metrics: Diccionario con métricas
        """
        self.info("Métricas de rendimiento", metrics=metrics)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtiene información del sistema
        
        Returns:
            Información del sistema
        """
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_id': os.getpid(),
            'thread_count': threading.active_count()
        }
    
    def log_system_info(self):
        """Registra información del sistema"""
        system_info = self.get_system_info()
        self.info("Información del sistema", system_info=system_info)
    
    def export_logs(self, output_path: str, format: str = "json"):
        """
        Exporta logs a archivo
        
        Args:
            output_path: Ruta del archivo de salida
            format: Formato de exportación (json, csv, txt)
        """
        try:
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'logs': self.memory_logs,
                        'counters': self.event_counters,
                        'performance_summary': self.performance_monitor.get_performance_summary(),
                        'system_info': self.get_system_info()
                    }, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == "csv":
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    if self.memory_logs:
                        writer = csv.DictWriter(f, fieldnames=self.memory_logs[0].keys())
                        writer.writeheader()
                        writer.writerows(self.memory_logs)
            
            elif format.lower() == "txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for log in self.memory_logs:
                        f.write(f"{log['timestamp']} | {log['level']} | {log['message']}\n")
            
            self.info(f"Logs exportados a: {output_path} (formato: {format})")
            
        except Exception as e:
            self.error(f"Error al exportar logs: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de logs
        
        Returns:
            Resumen de logs
        """
        return {
            'total_logs': len(self.memory_logs),
            'counters': self.event_counters,
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'system_info': self.get_system_info(),
            'recent_logs': self.memory_logs[-10:] if self.memory_logs else []
        }
    
    def clear_memory_logs(self):
        """Limpia los logs en memoria"""
        self.memory_logs.clear()
        self.info("Logs en memoria limpiados")
    
    def set_log_level(self, level: str):
        """
        Cambia el nivel de logging dinámicamente
        
        Args:
            level: Nuevo nivel de logging
        """
        self.log_level = level.upper()
        self._setup_loguru()
        self.info(f"Nivel de logging cambiado a: {self.log_level}")
    
    def log_configuration(self, config: Dict[str, Any]):
        """
        Registra la configuración del sistema
        
        Args:
            config: Configuración a registrar
        """
        self.info("Configuración del sistema", configuration=config)
    
    def create_checkpoint(self, checkpoint_name: str):
        """
        Crea un checkpoint de logs
        
        Args:
            checkpoint_name: Nombre del checkpoint
        """
        checkpoint = {
            'name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'logs_count': len(self.memory_logs),
            'counters': self.event_counters.copy(),
            'performance_summary': self.performance_monitor.get_performance_summary()
        }
        
        checkpoint_file = self.log_dir / f"checkpoint_{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        self.info(f"Checkpoint creado: {checkpoint_name}", checkpoint=checkpoint)
    
    def monitor_memory_usage(self):
        """Registra el uso de memoria actual"""
        memory = psutil.virtual_memory()
        self.debug("Uso de memoria", 
                  memory_percent=memory.percent,
                  memory_used_gb=memory.used / (1024**3),
                  memory_available_gb=memory.available / (1024**3))
    
    def monitor_cpu_usage(self):
        """Registra el uso de CPU actual"""
        cpu_percent = psutil.cpu_percent(interval=1)
        self.debug("Uso de CPU", cpu_percent=cpu_percent)


# Instancia global del logger
_global_logger = None

def get_logger() -> Logger:
    """
    Obtiene la instancia global del logger
    
    Returns:
        Instancia del logger
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger

def set_global_logger(logger_instance: Logger):
    """
    Establece la instancia global del logger
    
    Args:
        logger_instance: Instancia del logger
    """
    global _global_logger
    _global_logger = logger_instance
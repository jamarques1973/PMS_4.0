"""
PMS 4.0.0 - Pipeline Modeling Suite
===================================

Sistema modular de modelado de datos con arquitectura en capas.
Diseñado para ser totalmente compatible con Voilá y optimizado para máximo rendimiento.

Autor: jamarques1973
Versión: 4.0.0
"""

__version__ = "4.0.0"
__author__ = "jamarques1973"
__email__ = "jamarques1973@example.com"
__description__ = "Pipeline Modeling Suite 4.0.0 - Sistema modular de modelado de datos"

# Importaciones principales
from .main import PMSSystem
from .controller.config import Config
from .controller.logger import Logger

# Exponer las clases principales
__all__ = [
    "PMSSystem",
    "Config", 
    "Logger",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]

# Configuración inicial del sistema
def setup_logging():
    """Configura el sistema de logging por defecto"""
    from .controller.logger import Logger
    return Logger()

def load_config(config_path=None):
    """Carga la configuración del sistema"""
    from .controller.config import Config
    return Config(config_path)

# Inicialización automática
try:
    logger = setup_logging()
    config = load_config()
except Exception as e:
    print(f"Warning: No se pudo inicializar PMS 4.0.0 automáticamente: {e}")
    logger = None
    config = None
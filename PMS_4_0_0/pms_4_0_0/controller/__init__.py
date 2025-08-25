"""
Capa Controladora - Orquestación del Sistema PMS 4.0.0
=====================================================

Esta capa se encarga de:
- Coordinación entre capas Frontend y Backend
- Gestión de flujos de trabajo
- Manejo de errores y logging
- Configuración del sistema
"""

from .config import Config
from .logger import Logger
from .orchestrator import Orchestrator

__all__ = ["Config", "Logger", "Orchestrator"]
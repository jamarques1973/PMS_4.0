"""
Capa Frontend - Interfaz de Usuario PMS 4.0.0
============================================

Esta capa proporciona:
- Interfaz de usuario profesional y atractiva
- Widgets interactivos para configuración
- Visualizaciones y gráficos
- Sistema de ayuda integrado
"""

from .widgets import WidgetManager
from .themes import ThemeManager
from .layouts import LayoutManager

__all__ = ["WidgetManager", "ThemeManager", "LayoutManager"]
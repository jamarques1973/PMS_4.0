# PMS 4.0.0 - Pipeline Modeling Suite

## Descripción
PMS 4.0.0 es una refactorización completa del sistema PMS 3.6.0, transformando la arquitectura monolítica en una arquitectura modular y en capas. El sistema está diseñado para ser totalmente compatible con Voilá y optimizado para máximo rendimiento.

## Arquitectura de Capas

### 🎨 Capa Frontend (UI)
- **Ubicación**: `frontend/`
- **Tecnología**: IPyWidgets + HTML/CSS personalizado
- **Responsabilidades**:
  - Interfaz de usuario profesional y atractiva
  - Widgets interactivos para configuración
  - Visualizaciones y gráficos
  - Sistema de ayuda integrado

### ⚙️ Capa Backend (Lógica de Negocio)
- **Ubicación**: `backend/`
- **Responsabilidades**:
  - Procesamiento de datos
  - Entrenamiento de modelos
  - Optimización de hiperparámetros
  - Análisis de interpretabilidad
  - Generación de informes

### 🎯 Capa Controladora (Orquestación)
- **Ubicación**: `controller/`
- **Responsabilidades**:
  - Coordinación entre capas
  - Gestión de flujos de trabajo
  - Manejo de errores y logging
  - Configuración del sistema

## Estructura del Proyecto

```
PMS_4_0_0/
├── frontend/                 # Capa de interfaz de usuario
│   ├── widgets/             # Widgets personalizados
│   ├── themes/              # Temas y estilos
│   └── layouts/             # Layouts de interfaz
├── backend/                 # Capa de lógica de negocio
│   ├── data/               # Procesamiento de datos
│   ├── models/             # Modelos de ML
│   ├── optimization/       # Optimización de hiperparámetros
│   ├── xai/               # Interpretabilidad XAI
│   └── reporting/         # Generación de informes
├── controller/             # Capa de orquestación
│   ├── orchestrator.py    # Orquestador principal
│   ├── config.py          # Configuración del sistema
│   └── logger.py          # Sistema de logging
├── utils/                  # Utilidades compartidas
│   ├── decorators.py      # Decoradores comunes
│   ├── validators.py      # Validaciones
│   └── helpers.py         # Funciones auxiliares
├── config/                 # Configuraciones
│   ├── settings.py        # Configuración general
│   └── logging_config.py  # Configuración de logging
├── tests/                  # Tests unitarios
├── docs/                   # Documentación
├── examples/              # Ejemplos de uso
├── requirements.txt       # Dependencias
├── setup.py              # Instalación
└── main.py               # Punto de entrada principal
```

## Características Principales

### ✅ Objetivos Cumplidos

1. **Arquitectura Modular en Capas**: Separación clara entre Frontend, Backend y Controlador
2. **Compatibilidad Voilá**: Diseñado para ejecutarse sin problemas en Voilá
3. **Código Optimizado**: Eliminación de duplicaciones y uso de modelos únicos
4. **Reducción de Deuda Técnica**: Refactorización completa del código
5. **Máximo Rendimiento**: Optimización de tiempos de ejecución
6. **Comentarios Extensivos**: Documentación completa del código
7. **Sistema de Logging**: Trazas, logs y debug completos
8. **UI Profesional**: Interfaz atractiva e intuitiva
9. **Informes Formateados**: Salidas con experiencia de UI homogénea
10. **Máximo Ajuste**: Nivel máximo de personalización de parámetros

### 🚀 Nuevas Funcionalidades

- **Sistema de Plugins**: Arquitectura extensible para nuevos modelos
- **Pipeline Configurable**: Flujos de trabajo personalizables
- **Caché Inteligente**: Sistema de caché para optimizar rendimiento
- **Validación Robusta**: Validaciones en tiempo real
- **Exportación Múltiple**: Soporte para múltiples formatos de salida
- **Monitorización**: Sistema de monitorización en tiempo real

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/jamarques1973/PMS_4.0.git

# Instalar dependencias
pip install -r requirements.txt

# Configurar el sistema
python setup.py install
```

## Uso Rápido

```python
from pms_4_0_0.main import PMSSystem

# Inicializar el sistema
pms = PMSSystem()

# Ejecutar pipeline completo
pms.run_pipeline()

# O ejecutar módulos específicos
pms.data.load_data()
pms.models.train_all()
pms.xai.analyze_models()
pms.reporting.generate_report()
```

## Configuración

El sistema se configura a través de archivos YAML en el directorio `config/`:

```yaml
# config/settings.yaml
system:
  name: "PMS 4.0.0"
  version: "4.0.0"
  debug: true
  log_level: "INFO"

models:
  svr:
    enabled: true
    default_params:
      C: 1.0
      epsilon: 0.1
  neural_network:
    enabled: true
    default_params:
      layers: [64, 32]
      activation: "relu"
  xgboost:
    enabled: true
    default_params:
      n_estimators: 100
      learning_rate: 0.1
```

## Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

- **Autor**: jamarques1973
- **Email**: [email del autor]
- **Proyecto**: https://github.com/jamarques1973/PMS_4.0

## Changelog

### v4.0.0
- Refactorización completa de la arquitectura
- Implementación de arquitectura en capas
- Optimización de rendimiento
- Nuevo sistema de logging
- UI completamente rediseñada
- Sistema de plugins
- Compatibilidad total con Voilá
# PMS 4.0.0 - Pipeline Modeling Suite
## Documentación Completa

### 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Instalación](#instalación)
4. [Configuración](#configuración)
5. [Uso Básico](#uso-básico)
6. [Componentes del Sistema](#componentes-del-sistema)
7. [API de Referencia](#api-de-referencia)
8. [Ejemplos](#ejemplos)
9. [Despliegue en Voilá](#despliegue-en-voilá)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## 🎯 Introducción

PMS 4.0.0 es una refactorización completa del sistema PMS 3.6.0, transformando la arquitectura monolítica en una arquitectura modular y en capas. El sistema está diseñado para ser totalmente compatible con Voilá y optimizado para máximo rendimiento.

### ✅ Objetivos Cumplidos

1. **Arquitectura Modular y en Capas**: Separación clara entre Frontend, Backend y Controlador
2. **Compatibilidad con Voilá**: Totalmente compatible para ejecución web
3. **Código Optimizado**: Reutilización y optimización del código
4. **Reducción de Deuda Técnica**: Código limpio y bien estructurado
5. **Máximo Rendimiento**: Procesamiento paralelo y optimización de memoria
6. **Documentación Completa**: Comentarios detallados y documentación
7. **Sistema de Logs**: Trazas, logs y debug completos
8. **UI Profesional**: Interfaz moderna y atractiva
9. **Informes Formateados**: Reportes profesionales y homogéneos
10. **Máximo Ajuste**: Configuración flexible y personalizable

---

## 🏗️ Arquitectura del Sistema

### Estructura de Capas

```
PMS 4.0.0
├── 🎨 Frontend (UI)
│   ├── Widgets
│   ├── Themes
│   └── Layouts
├── ⚙️ Backend (Lógica de Negocio)
│   ├── Data Processing
│   ├── Models
│   ├── Optimization
│   ├── XAI
│   └── Reporting
└── 🎛️ Controller (Orquestación)
    ├── Configuration
    ├── Logging
    └── Orchestrator
```

### Componentes Principales

#### 🎨 Capa Frontend
- **Widgets**: Componentes interactivos para configuración
- **Themes**: Temas visuales personalizables
- **Layouts**: Diferentes disposiciones de la interfaz

#### ⚙️ Capa Backend
- **Data Processing**: Carga, validación y preprocesamiento de datos
- **Models**: Entrenamiento y evaluación de modelos
- **Optimization**: Optimización de hiperparámetros
- **XAI**: Análisis de interpretabilidad
- **Reporting**: Generación de informes

#### 🎛️ Capa Controlador
- **Configuration**: Gestión centralizada de configuración
- **Logging**: Sistema de logs y trazas
- **Orchestrator**: Coordinación entre capas

---

## 📦 Instalación

### Requisitos del Sistema

- Python 3.8+
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio en disco

### Instalación desde PyPI

```bash
pip install pms-4-0-0
```

### Instalación desde Fuente

```bash
git clone https://github.com/jamarques1973/PMS_4.0.git
cd PMS_4.0
pip install -r requirements.txt
pip install -e .
```

### Verificación de Instalación

```python
from pms_4_0_0 import PMSSystem
print("✅ PMS 4.0.0 instalado correctamente")
```

---

## ⚙️ Configuración

### Configuración Básica

```python
from pms_4_0_0.controller.config import Config

# Crear configuración por defecto
config = Config()

# Personalizar parámetros
config.system.log_level = "INFO"
config.system.max_workers = 4
config.data.test_size = 0.2
config.models.svr.enabled = True
config.optimization.enabled = True
```

### Parámetros de Configuración

#### Sistema
- `log_level`: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
- `max_workers`: Número máximo de workers paralelos
- `cache_enabled`: Habilitar caché
- `output_dir`: Directorio de salida

#### Datos
- `test_size`: Proporción de datos de prueba
- `random_state`: Semilla para reproducibilidad
- `validation.missing_threshold`: Umbral para valores faltantes
- `validation.outlier_threshold`: Umbral para outliers

#### Modelos
- `svr.enabled`: Habilitar SVR
- `neural_network.enabled`: Habilitar Red Neuronal
- `xgboost.enabled`: Habilitar XGBoost
- `random_forest.enabled`: Habilitar Random Forest
- `rnn.enabled`: Habilitar RNN

#### Optimización
- `enabled`: Habilitar optimización
- `max_trials`: Número máximo de trials
- `timeout`: Timeout en segundos

#### XAI
- `enabled`: Habilitar análisis XAI
- `methods.shap.enabled`: Habilitar SHAP
- `methods.lime.enabled`: Habilitar LIME

---

## 🚀 Uso Básico

### Inicialización del Sistema

```python
from pms_4_0_0 import PMSSystem

# Inicializar con configuración por defecto
pms = PMSSystem()

# O inicializar con configuración personalizada
config = Config()
# ... configurar parámetros ...
pms = PMSSystem(config)
```

### Pipeline Completo

```python
# Ejecutar pipeline completo
result = pms.run_pipeline(
    pipeline_name="mi_pipeline",
    data=mi_dataframe,
    target_column='target',
    test_size=0.2
)

# Verificar resultados
for step_name, step_result in result.items():
    print(f"{step_name}: {step_result['status']}")
```

### Uso Modular

```python
# Cargar datos
data_info = pms.data.load_data("datos.csv")

# Preprocesar datos
preprocessed_data = pms.data.preprocess_data(
    data_key="loaded_data",
    missing_strategy="impute",
    scaling_method="standard"
)

# Entrenar modelos
training_result = pms.models.train_all_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# Evaluar modelos
evaluation_result = pms.models.evaluate_all_models(
    X_test=X_test,
    y_test=y_test
)

# Optimizar hiperparámetros
optimization_result = pms.optimization.optimize_all_models(
    X_train=X_train,
    y_train=y_train,
    max_trials=50
)

# Análisis XAI
xai_result = pms.xai.analyze_all_models(
    X_test=X_test,
    y_test=y_test
)

# Generar informe
report_result = pms.reporting.generate_comprehensive_report(
    data_info=data_info,
    model_results={
        'training': training_result,
        'evaluation': evaluation_result,
        'optimization': optimization_result,
        'xai': xai_result
    }
)
```

---

## 🔧 Componentes del Sistema

### Data Processing

#### DataLoader
```python
# Cargar archivo
data_info = pms.data.loader.load_file("datos.csv")

# Cargar múltiples archivos
data_info = pms.data.loader.load_multiple_files([
    "datos1.csv",
    "datos2.csv"
])

# Validar formato
validation = pms.data.loader.validate_file_format("datos.csv")
```

#### DataValidator
```python
# Validar dataset
validation_result = pms.data.validator.validate_dataset(data)

# Validar datos preprocesados
validation_result = pms.data.validator.validate_preprocessed_data(data)

# Obtener reporte de validación
report = pms.data.validator.get_validation_report(validation_result)
```

#### DataPreprocessor
```python
# Preprocesar datos
preprocessed_data = pms.data.preprocessor.preprocess(
    data,
    missing_strategy="impute",
    scaling_method="standard",
    encoding_method="auto"
)

# Aplicar transformaciones a nuevos datos
transformed_data = pms.data.preprocessor.transform(new_data)

# Revertir transformaciones
original_data = pms.data.preprocessor.inverse_transform(transformed_data)
```

#### FeatureEngineer
```python
# Crear características
feature_data = pms.data.feature_engineer.create_features(
    data,
    polynomial_features=True,
    interaction_features=True,
    temporal_features=True
)

# Seleccionar características
selection_result = pms.data.feature_engineer.select_features(
    data,
    target=target,
    method="auto",
    n_features=20
)

# Obtener importancia de características
importance = pms.data.feature_engineer.get_feature_importance(
    data,
    target,
    method="auto"
)
```

#### ExploratoryAnalyzer
```python
# Análisis exploratorio completo
analysis_result = pms.data.exploratory_analyzer.analyze_dataset(data)

# Crear visualizaciones
plot_paths = pms.data.exploratory_analyzer.create_visualizations(
    data,
    output_dir="./output/plots"
)

# Generar reporte
report_path = pms.data.exploratory_analyzer.generate_report(
    analysis_result,
    output_path="./output/report.html"
)
```

### Models

#### Entrenamiento
```python
# Entrenar todos los modelos
training_result = pms.models.train_all_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# Entrenar modelo específico
result = pms.models.train_model(
    model_name="svr",
    X_train=X_train,
    y_train=y_train
)
```

#### Evaluación
```python
# Evaluar todos los modelos
evaluation_result = pms.models.evaluate_all_models(
    X_test=X_test,
    y_test=y_test
)

# Evaluar modelo específico
metrics = pms.models.evaluate_model(
    model_name="svr",
    X_test=X_test,
    y_test=y_test
)
```

#### Predicción
```python
# Predecir con todos los modelos
predictions = pms.models.predict_all_models(X_new)

# Predecir con modelo específico
prediction = pms.models.predict_model(
    model_name="svr",
    X_new=X_new
)
```

### Optimization

```python
# Optimizar todos los modelos
optimization_result = pms.optimization.optimize_all_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    max_trials=50,
    timeout=300
)

# Optimizar modelo específico
result = pms.optimization.optimize_model(
    model_name="svr",
    X_train=X_train,
    y_train=y_train,
    max_trials=20
)
```

### XAI (Explainable AI)

```python
# Análisis XAI completo
xai_result = pms.xai.analyze_all_models(
    X_test=X_test,
    y_test=y_test,
    sample_size=100
)

# Análisis XAI para modelo específico
result = pms.xai.analyze_model(
    model_name="svr",
    X_test=X_test,
    y_test=y_test,
    methods=["shap", "lime"]
)
```

### Reporting

```python
# Generar informe completo
report_result = pms.reporting.generate_comprehensive_report(
    data_info=data_info,
    model_results=model_results,
    output_dir="./output/reports"
)

# Generar informe específico
report_path = pms.reporting.generate_model_report(
    model_name="svr",
    data_info=data_info,
    model_results=model_results
)
```

---

## 📚 API de Referencia

### PMSSystem

#### Métodos Principales

```python
class PMSSystem:
    def __init__(self, config_path: Optional[str] = None)
    def run_pipeline(self, pipeline_name: str, **kwargs) -> Dict[str, Any]
    def run_single_step(self, step_name: str, **kwargs) -> Dict[str, Any]
    def get_status(self) -> Dict[str, Any]
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]
    def create_custom_pipeline(self, name: str, steps: List[str]) -> str
    def load_custom_pipeline(self, name: str) -> Dict[str, Any]
    def export_logs(self, output_path: str, format: str = "json")
    def get_log_summary(self) -> Dict[str, Any]
    def update_config(self, section: str, key: str, value: Any)
    def save_config(self, output_path: str)
    def shutdown(self)
```

#### Propiedades

- `config`: Configuración del sistema
- `logger`: Sistema de logging
- `data`: Procesador de datos
- `models`: Gestor de modelos
- `optimization`: Motor de optimización
- `xai`: Analizador XAI
- `reporting`: Generador de reportes
- `widgets`: Gestor de widgets
- `themes`: Gestor de temas
- `layouts`: Gestor de layouts

### Config

```python
class Config:
    def __init__(self, config_path: Optional[str] = None)
    def validate_config(self) -> bool
    def get_all_config(self) -> Dict[str, Any]
    def update_config(self, section: str, key: str, value: Any)
    def save_config(self, output_path: str)
    def load_config(self, config_path: str)
```

### Logger

```python
class Logger:
    def __init__(self, log_level: str = "INFO", log_dir: str = "./logs")
    def info(self, message: str)
    def warning(self, message: str)
    def error(self, message: str)
    def debug(self, message: str)
    def log_exception(self, exception: Exception, context: str)
    def operation_trace(self, operation_name: str)
    def export_logs(self, output_path: str, format: str = "json")
    def get_log_summary(self) -> Dict[str, Any]
```

---

## 💡 Ejemplos

### Ejemplo 1: Pipeline Básico

```python
from pms_4_0_0 import PMSSystem
import pandas as pd

# Crear datos de ejemplo
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'target': np.random.randn(1000)
})

# Inicializar sistema
pms = PMSSystem()

# Ejecutar pipeline
result = pms.run_pipeline(
    pipeline_name="ejemplo_basico",
    data=data,
    target_column='target'
)

print("Pipeline completado:", result)
```

### Ejemplo 2: Configuración Personalizada

```python
from pms_4_0_0 import PMSSystem
from pms_4_0_0.controller.config import Config

# Crear configuración personalizada
config = Config()
config.system.log_level = "DEBUG"
config.models.svr.enabled = True
config.models.xgboost.enabled = True
config.optimization.enabled = True
config.optimization.max_trials = 30

# Inicializar sistema con configuración
pms = PMSSystem(config)

# Ejecutar pipeline
result = pms.run_pipeline(
    pipeline_name="configurado",
    data=data,
    target_column='target'
)
```

### Ejemplo 3: Uso Modular

```python
from pms_4_0_0 import PMSSystem

pms = PMSSystem()

# Cargar y preprocesar datos
data_info = pms.data.load_data("datos.csv")
preprocessed_data = pms.data.preprocess_data(
    data_key="loaded_data",
    missing_strategy="impute",
    scaling_method="standard"
)

# Dividir datos
split_result = pms.data.split_data(
    data_key="preprocessed_data",
    test_size=0.2
)

# Entrenar modelos
training_result = pms.models.train_all_models(
    X_train=split_result['train_data'].drop(columns=['target']),
    y_train=split_result['train_data']['target'],
    X_test=split_result['test_data'].drop(columns=['target']),
    y_test=split_result['test_data']['target']
)

# Evaluar modelos
evaluation_result = pms.models.evaluate_all_models(
    X_test=split_result['test_data'].drop(columns=['target']),
    y_test=split_result['test_data']['target']
)

# Generar informe
report_result = pms.reporting.generate_comprehensive_report(
    data_info=data_info,
    model_results={
        'training': training_result,
        'evaluation': evaluation_result
    }
)
```

---

## 🌐 Despliegue en Voilá

### Instalación de Voilá

```bash
pip install voila
```

### Crear Notebook para Voilá

```python
# notebook.ipynb
import ipywidgets as widgets
from pms_4_0_0 import PMSSystem

# Crear widgets de interfaz
upload_widget = widgets.FileUpload(
    accept='.csv,.xlsx',
    multiple=False,
    description='Subir Datos'
)

target_widget = widgets.Dropdown(
    options=[],
    description='Variable Objetivo'
)

run_button = widgets.Button(
    description='Ejecutar Pipeline',
    button_style='success'
)

output_widget = widgets.Output()

# Función para ejecutar pipeline
def run_pipeline(b):
    with output_widget:
        output_widget.clear_output()
        
        # Inicializar sistema
        pms = PMSSystem()
        
        # Cargar datos
        data = pd.read_csv(upload_widget.value[0]['content'])
        
        # Actualizar opciones de variable objetivo
        target_widget.options = list(data.columns)
        
        # Ejecutar pipeline
        result = pms.run_pipeline(
            pipeline_name="voila_pipeline",
            data=data,
            target_column=target_widget.value
        )
        
        print("Pipeline completado:", result)

run_button.on_click(run_pipeline)

# Mostrar widgets
widgets.VBox([
    upload_widget,
    target_widget,
    run_button,
    output_widget
])
```

### Ejecutar Voilá

```bash
voila notebook.ipynb --port=8866
```

### Configuración de Voilá

```bash
# Configuración avanzada
voila notebook.ipynb \
    --port=8866 \
    --host=0.0.0.0 \
    --no-browser \
    --enable_nbextensions=True \
    --VoilaConfiguration.template=default
```

---

## 🔧 Troubleshooting

### Problemas Comunes

#### Error de Importación
```
ModuleNotFoundError: No module named 'pms_4_0_0'
```

**Solución:**
```bash
pip install pms-4-0-0
# o
pip install -e .
```

#### Error de Memoria
```
MemoryError: Unable to allocate array
```

**Solución:**
```python
# Reducir tamaño de datos
config.data.max_rows = 10000
config.system.max_workers = 2
```

#### Error de Dependencias
```
ImportError: No module named 'sklearn'
```

**Solución:**
```bash
pip install -r requirements.txt
```

#### Error de Configuración
```
ValueError: Configuración inválida
```

**Solución:**
```python
# Validar configuración
if not config.validate_config():
    print("Errores de configuración:", config.get_validation_errors())
```

### Logs y Debug

```python
# Habilitar logs detallados
config.system.log_level = "DEBUG"

# Exportar logs
pms.export_logs("debug_logs.json")

# Obtener resumen de logs
log_summary = pms.get_log_summary()
print("Errores:", log_summary['error_logs'])
```

### Verificación del Sistema

```python
# Verificar estado del sistema
status = pms.get_status()
health = pms.orchestrator.get_system_health()

print("Estado:", status['system_info']['status'])
print("Componentes:", health['components'])
```

---

## ❓ FAQ

### P: ¿PMS 4.0.0 es compatible con PMS 3.6.0?

**R:** PMS 4.0.0 es una refactorización completa y no es directamente compatible con la versión 3.6.0. Sin embargo, mantiene todas las funcionalidades principales con una API mejorada.

### P: ¿Cómo migrar de PMS 3.6.0 a PMS 4.0.0?

**R:** 
1. Instalar PMS 4.0.0
2. Adaptar el código usando la nueva API modular
3. Configurar parámetros usando el nuevo sistema de configuración
4. Ejecutar y verificar resultados

### P: ¿PMS 4.0.0 es más lento que PMS 3.6.0?

**R:** No, PMS 4.0.0 incluye optimizaciones significativas:
- Procesamiento paralelo
- Caché inteligente
- Optimización de memoria
- Reducción de operaciones redundantes

### P: ¿Puedo usar PMS 4.0.0 sin interfaz gráfica?

**R:** Sí, PMS 4.0.0 puede usarse completamente desde línea de comandos o scripts Python sin necesidad de interfaz gráfica.

### P: ¿Qué formatos de datos soporta PMS 4.0.0?

**R:** PMS 4.0.0 soporta múltiples formatos:
- CSV
- Excel (.xlsx, .xls)
- JSON
- Parquet
- Pickle
- YAML
- Texto plano

### P: ¿Puedo agregar mis propios modelos?

**R:** Sí, PMS 4.0.0 está diseñado para ser extensible. Puedes agregar modelos personalizados implementando la interfaz estándar.

### P: ¿Cómo configurar el procesamiento paralelo?

**R:** 
```python
config.system.max_workers = 4  # Número de workers
config.system.cache_enabled = True  # Habilitar caché
```

### P: ¿PMS 4.0.0 soporta GPU?

**R:** PMS 4.0.0 soporta GPU a través de TensorFlow y XGBoost cuando están disponibles. La detección es automática.

### P: ¿Cómo personalizar los reportes?

**R:** Los reportes se pueden personalizar modificando las plantillas HTML y CSS en el módulo de reporting.

### P: ¿PMS 4.0.0 es compatible con Jupyter?

**R:** Sí, PMS 4.0.0 es totalmente compatible con Jupyter Notebooks y JupyterLab.

---

## 📞 Soporte

### Recursos Adicionales

- **Documentación**: [GitHub Wiki](https://github.com/jamarques1973/PMS_4.0/wiki)
- **Issues**: [GitHub Issues](https://github.com/jamarques1973/PMS_4.0/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/jamarques1973/PMS_4.0/discussions)

### Contacto

- **Autor**: jamarques1973
- **Email**: jamarques1973@example.com
- **GitHub**: [@jamarques1973](https://github.com/jamarques1973)

---

## 📄 Licencia

PMS 4.0.0 está licenciado bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

---

## 🎉 Agradecimientos

Gracias a todos los contribuyentes y usuarios que han hecho posible esta refactorización completa de PMS 4.0.0.

---

*Última actualización: Diciembre 2024*
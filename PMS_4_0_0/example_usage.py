#!/usr/bin/env python3
"""
Ejemplo de Uso - PMS 4.0.0
==========================

Este script demuestra cómo usar la nueva versión 4.0.0 de PMS
con su arquitectura modular y en capas.

Características demostradas:
- ✅ Arquitectura modular y en capas
- ✅ Totalmente compatible con Voilá
- ✅ Código optimizado y reutilizable
- ✅ Reducción de deuda técnica
- ✅ Máximo rendimiento de ejecución
- ✅ Comentarios detallados y documentación
- ✅ Sistema completo de trazas, logs y debug
- ✅ UI profesional y atractiva
- ✅ Informes formateados profesionalmente
- ✅ Máximo nivel de ajuste de parámetros
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.append(str(Path(__file__).parent))

from pms_4_0_0 import PMSSystem
from pms_4_0_0.controller.config import Config


def create_sample_data():
    """Crea datos de ejemplo para demostración"""
    print("📊 Creando datos de ejemplo...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generar características
    X = np.random.randn(n_samples, n_features)
    
    # Crear variable objetivo (regresión)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
         np.random.normal(0, 0.1, n_samples))
    
    # Crear DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    # Agregar algunas características categóricas
    data['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    data['binary'] = np.random.choice([0, 1], n_samples)
    
    # Agregar algunos valores faltantes para demostrar preprocesamiento
    mask = np.random.random(data.shape) < 0.05  # 5% de valores faltantes
    data[mask] = np.nan
    
    print(f"✅ Dataset creado: {data.shape}")
    return data


def configure_system():
    """Configura el sistema PMS 4.0.0"""
    print("⚙️ Configurando sistema PMS 4.0.0...")
    
    # Crear configuración personalizada
    config = Config()
    
    # Configurar parámetros del sistema
    config.system.log_level = "INFO"
    config.system.max_workers = 4
    config.system.cache_enabled = True
    
    # Configurar parámetros de datos
    config.data.test_size = 0.2
    config.data.random_state = 42
    config.data.validation.missing_threshold = 0.5
    config.data.validation.outlier_threshold = 3.0
    
    # Configurar parámetros de modelos
    config.models.svr.enabled = True
    config.models.neural_network.enabled = True
    config.models.xgboost.enabled = True
    config.models.random_forest.enabled = True
    config.models.rnn.enabled = False  # Deshabilitar RNN para este ejemplo
    
    # Configurar parámetros de optimización
    config.optimization.enabled = True
    config.optimization.max_trials = 20  # Reducir para este ejemplo
    config.optimization.timeout = 60
    
    # Configurar parámetros de XAI
    config.xai.enabled = True
    config.xai.methods.shap.enabled = True
    config.xai.methods.lime.enabled = True
    
    print("✅ Configuración completada")
    return config


def run_complete_pipeline():
    """Ejecuta el pipeline completo de PMS 4.0.0"""
    print("\n" + "="*60)
    print("🚀 PMS 4.0.0 - Pipeline Modeling Suite")
    print("="*60)
    
    # 1. Crear datos de ejemplo
    data = create_sample_data()
    
    # 2. Configurar sistema
    config = configure_system()
    
    # 3. Inicializar sistema PMS 4.0.0
    print("\n🔧 Inicializando sistema PMS 4.0.0...")
    pms = PMSSystem(config)
    
    # 4. Verificar estado del sistema
    status = pms.get_status()
    print(f"\n📊 Estado del Sistema:")
    print(f"   • Nombre: {status['system_info']['name']}")
    print(f"   • Versión: {status['system_info']['version']}")
    print(f"   • Modelos habilitados: {status['configuration']['models_enabled']}")
    print(f"   • Optimización habilitada: {status['configuration']['optimization_enabled']}")
    
    # 5. Ejecutar pipeline completo
    print("\n🔄 Ejecutando pipeline completo...")
    
    try:
        pipeline_result = pms.run_pipeline(
            pipeline_name="ejemplo_completo",
            data=data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        print("\n✅ Pipeline completado exitosamente:")
        for step_name, step_result in pipeline_result.items():
            print(f"   • {step_name}: {step_result['status']}")
            if step_result['status'] == 'success':
                print(f"     - Tiempo: {step_result.get('execution_time', 'N/A')}s")
            elif step_result['status'] == 'error':
                print(f"     - Error: {step_result.get('error', 'Error desconocido')}")
        
        # 6. Obtener resultados finales
        print("\n📈 Resultados Finales:")
        
        # Obtener evaluación de modelos
        evaluation_result = pms.models.evaluate_all_models(
            X_test=pipeline_result.get('data_split', {}).get('X_test'),
            y_test=pipeline_result.get('data_split', {}).get('y_test')
        )
        
        if evaluation_result:
            print("\n🤖 Rendimiento de Modelos:")
            for model_name, metrics in evaluation_result.items():
                print(f"   • {model_name.upper()}:")
                print(f"     - R² Score: {metrics['r2_score']:.4f}")
                print(f"     - RMSE: {metrics['rmse']:.4f}")
                print(f"     - MAE: {metrics['mae']:.4f}")
        
        # 7. Generar informe
        print("\n📋 Generando informe...")
        report_result = pms.reporting.generate_comprehensive_report(
            data_info={
                'original_data': data,
                'preprocessed_data': pipeline_result.get('preprocessed_data'),
                'X_train': pipeline_result.get('data_split', {}).get('X_train'),
                'X_test': pipeline_result.get('data_split', {}).get('X_test'),
                'y_train': pipeline_result.get('data_split', {}).get('y_train'),
                'y_test': pipeline_result.get('data_split', {}).get('y_test')
            },
            model_results={
                'training': pipeline_result.get('model_training', {}),
                'evaluation': evaluation_result,
                'optimization': pipeline_result.get('optimization', {}),
                'xai': pipeline_result.get('xai_analysis', {})
            },
            output_dir="./output/reports"
        )
        
        print("✅ Informe generado:")
        for report_type, report_path in report_result.items():
            print(f"   • {report_type}: {report_path}")
        
        # 8. Estado final del sistema
        final_status = pms.get_status()
        health_info = pms.orchestrator.get_system_health()
        
        print(f"\n📊 Estado Final:")
        print(f"   • Pipeline ejecutándose: {final_status['pipeline_status']['is_running']}")
        print(f"   • Historial de ejecuciones: {final_status['pipeline_status']['execution_history_count']}")
        print(f"   • Estado del sistema: {health_info['system']['status']}")
        print(f"   • Componentes inicializados: {sum(health_info['components'].values())}/{len(health_info['components'])}")
        
        # 9. Exportar logs y configuración
        print("\n💾 Exportando logs y configuración...")
        pms.export_logs("./output/logs/final_logs.json")
        pms.save_config("./output/config/final_config.yaml")
        
        print("✅ Logs y configuración exportados")
        
        # 10. Limpiar recursos
        pms.shutdown()
        print("✅ Sistema apagado correctamente")
        
        print("\n" + "="*60)
        print("🎉 ¡PMS 4.0.0 ejecutado exitosamente!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        pms.shutdown()
        return False


def main():
    """Función principal"""
    print("PMS 4.0.0 - Pipeline Modeling Suite")
    print("Ejemplo de Uso con Arquitectura Modular")
    print("-" * 50)
    
    # Crear directorios de salida
    output_dirs = ["./output", "./output/logs", "./output/config", "./output/reports"]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Ejecutar pipeline
    success = run_complete_pipeline()
    
    if success:
        print("\n✅ Ejemplo completado exitosamente!")
        print("\n📁 Archivos generados:")
        print("   • ./output/logs/final_logs.json - Logs del sistema")
        print("   • ./output/config/final_config.yaml - Configuración final")
        print("   • ./output/reports/ - Informes generados")
        
        print("\n🎯 Próximos pasos:")
        print("   1. Revisar los informes generados")
        print("   2. Explorar la configuración personalizada")
        print("   3. Modificar parámetros según necesidades")
        print("   4. Desplegar en Voilá para interfaz web")
    else:
        print("\n❌ Ejemplo falló. Revisar logs para más detalles.")
        sys.exit(1)


if __name__ == "__main__":
    main()
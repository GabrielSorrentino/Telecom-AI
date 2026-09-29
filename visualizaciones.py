import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from modelo_service import preparar_datos_ml

def mostrar_matriz_correlacion(X, y):
    """Muestra la matriz de correlación de las variables"""
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
    sns.set_theme(style='white')
    corr_matrix = df_temp.corr()
    
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        annot_kws={'size': 10}
    )
    plt.title('Matriz de Correlación de las Variables', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print('Correlación con la Cancelación (churn):')
    print(corr_matrix['canceloServicio'].sort_values(ascending=False))
    
    return corr_matrix

def boxplot_estandarizado(X, y, columna_floats, titulo=None):
    """Genera un boxplot de una variable numérica estandarizada vs cancelación"""
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='canceloServicio', y=columna_floats, data=df_temp, palette='Set2')
    
    if titulo is None:
        titulo = f'Relación entre {columna_floats} (Estandarizado) y Cancelación'
    plt.title(titulo)
    plt.xlabel('¿Canceló? (0 = No, 1 = Sí)')
    plt.ylabel(f'{columna_floats} (Escalado)')
    plt.show()

def boxplot_cuentas_mensuales(X, y):
    """Boxplot de cuentas mensuales vs cancelación"""
    return boxplot_estandarizado(X, y, 'cuentasMensuales', 'Relación entre Cuentas Mensuales (Estandarizado) y Cancelación')

def boxplot_antiguedad(X, y):
    """Boxplot de antigüedad vs cancelación"""
    return boxplot_estandarizado(X, y, 'antiguedadEnMeses', 'Relación entre Antigüedad (Estandarizado) y Cancelación')

def scatter_antiguedad_vs_cuentas(X, y):
    """Diagrama de dispersión de antigüedad vs cuentas mensuales coloreado por cancelación"""
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='antiguedadEnMeses', 
        y='cuentasMensuales',
        hue=df_temp['canceloServicio'].map({0: 'No', 1: 'Sí'}), 
        data=df_temp, 
        alpha=0.5
    )
    plt.title('Tendencia de Cancelación: Antigüedad vs. Cuentas Mensuales')
    plt.xlabel('Antigüedad (meses)')
    plt.ylabel('Cuentas mensuales')
    plt.legend(title='Cancelación')
    plt.show()

def histograma_cancelacion_general():
    """Histograma general de distribución de cancelación"""
    from estadisticas import distribucion_cancelacion
    return distribucion_cancelacion()

def graficos_categoricos_cancelacion():
    """Genera múltiples gráficos categóricos de cancelación"""
    from estadisticas import (
        grafico_cancelacion_por_genero,
        grafico_cancelacion_por_contrato,
        grafico_cancelacion_por_pago,
        grafico_cancelacion_por_antiguedad,
        grafico_cancelacion_por_jubilado,
        grafico_cancelacion_por_internet
    )
    
    graficos = {
        'genero': grafico_cancelacion_por_genero(),
        'contrato': grafico_cancelacion_por_contrato(),
        'pago': grafico_cancelacion_por_pago(),
        'antiguedad': grafico_cancelacion_por_antiguedad(),
        'jubilado': grafico_cancelacion_por_jubilado(),
        'internet': grafico_cancelacion_por_internet()
    }
    
    return graficos

def graficos_costos_cancelacion():
    """Genera gráficos de costos vs cancelación"""
    from estadisticas import grafico_cancelacion_por_costo_mensual, grafico_cancelacion_por_cobro_total
    
    graficos = {
        'costo_mensual': grafico_cancelacion_por_costo_mensual(),
        'cobro_total': grafico_cancelacion_por_cobro_total()
    }
    
    return graficos

def mostrar_importancia_variables(model, X, nombre_modelo):
    """Muestra la importancia de las variables para el modelo"""
    if hasattr(model, 'feature_importances_'):
        # Para Random Forest
        importancia = pd.DataFrame({
            'Variable': X.columns,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importancia', y='Variable', data=importancia.head(10))
        plt.title(f'Importancia de Variables - {nombre_modelo}')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.show()
        
        print(f"\nImportancia de variables - {nombre_modelo}:")
        print(importancia)
        
    elif hasattr(model, 'coef_'):
        # Para Regresión Logística
        coeficientes = pd.DataFrame({
            'Variable': X.columns,
            'Coeficiente': model.coef_[0]
        }).sort_values('Coeficiente', key=abs, ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coeficiente', y='Variable', data=coeficientes.head(10))
        plt.title(f'Coeficientes del Modelo - {nombre_modelo}')
        plt.xlabel('Coeficiente')
        plt.tight_layout()
        plt.show()
        
        print(f"\nCoeficientes del modelo - {nombre_modelo}:")
        print(coeficientes)
    
    return importancia if hasattr(model, 'feature_importances_') else coeficientes

def matriz_confusion_visual(y_test, y_pred, nombre_modelo):
    """Muestra una matriz de confusión visual"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.xticks([0.5, 1.5], ['No Canceló', 'Canceló'])
    plt.yticks([0.5, 1.5], ['No Canceló', 'Canceló'])
    plt.show()
    
    return cm

def dashboard_completo():
    """Genera un dashboard completo con todas las visualizaciones principales"""
    print("=== Generando Dashboard Completo ===\n")
    
    # Preparar datos
    X, y = preparar_datos_ml()
    
    # 1. Matriz de correlación
    print("1. Matriz de Correlación")
    mostrar_matriz_correlacion(X, y)
    
    # 2. Distribución general de cancelación
    print("\n2. Distribución General de Cancelación")
    histograma_cancelacion_general()
    
    # 3. Boxplots
    print("\n3. Boxplots de Variables Numéricas")
    boxplot_cuentas_mensuales(X, y)
    boxplot_antiguedad(X, y)
    
    # 4. Scatter plot
    print("\n4. Scatter Plot: Antigüedad vs Cuentas Mensuales")
    scatter_antiguedad_vs_cuentas(X, y)
    
    # 5. Gráficos categóricos
    print("\n5. Gráficos Categóricos")
    graficos_cat = graficos_categoricos_cancelacion()
    
    # 6. Gráficos de costos
    print("\n6. Gráficos de Costos")
    graficos_cost = graficos_costos_cancelacion()
    
    return {
        'X': X,
        'y': y,
        'graficos_categoricos': graficos_cat,
        'graficos_costos': graficos_cost
    }
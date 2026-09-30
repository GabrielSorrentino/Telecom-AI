import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modelo_service import preparar_datos_ml
from nombres_variables import renombrar_columnas_dataframe

def mostrar_matriz_correlacion(X, y):
    """Genera la matriz de correlación de las variables"""
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
    # Renombrar columnas a nombres amigables (variables procesadas)
    df_temp = renombrar_columnas_dataframe(df_temp, es_procesado=True)
    
    sns.set_theme(style='white')
    corr_matrix = df_temp.corr()
    
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        annot_kws={'size': 10},
        ax=ax
    )
    ax.set_title('Matriz de Correlación de las Variables', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    print('Correlación con la Cancelación (churn):')
    print(corr_matrix['Canceló el servicio'].sort_values(ascending=False))
    
    return fig, corr_matrix

def boxplot_estandarizado(X, y, columna_floats, titulo=None):
    """Genera un boxplot de una variable numérica estandarizada vs cancelación"""
    from nombres_variables import obtener_nombre_amigable
    
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
    # Obtener nombre amigable para la columna
    nombre_amigable = obtener_nombre_amigable(columna_floats, es_procesada=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='canceloServicio', y=columna_floats, data=df_temp, palette='Set2', ax=ax)
    
    if titulo is None:
        titulo = f'Relación entre {nombre_amigable} (Estandarizado) y Cancelación'
    ax.set_title(titulo)
    ax.set_xlabel('¿Canceló? (0 = No, 1 = Sí)')
    ax.set_ylabel(f'{nombre_amigable} (Escalado)')
    plt.tight_layout()
    
    return fig

def boxplot_cuentas_mensuales(X, y):
    """Boxplot de cuentas mensuales vs cancelación"""
    from nombres_variables import obtener_nombre_amigable
    nombre_amigable = obtener_nombre_amigable('cuentasMensuales', es_procesada=True)
    return boxplot_estandarizado(X, y, 'cuentasMensuales', f'Relación entre {nombre_amigable} (Estandarizado) y Cancelación')

def boxplot_antiguedad(X, y):
    """Boxplot de antigüedad vs cancelación"""
    from nombres_variables import obtener_nombre_amigable
    nombre_amigable = obtener_nombre_amigable('antiguedadEnMeses', es_procesada=True)
    return boxplot_estandarizado(X, y, 'antiguedadEnMeses', f'Relación entre {nombre_amigable} (Estandarizado) y Cancelación')

def scatter_antiguedad_vs_cuentas(X, y):
    """Diagrama de dispersión de antigüedad vs cuentas mensuales coloreado por cancelación"""
    from nombres_variables import obtener_nombre_amigable
    
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
    # Obtener nombres amigables
    nombre_antiguedad = obtener_nombre_amigable('antiguedadEnMeses', es_procesada=True)
    nombre_cuentas = obtener_nombre_amigable('cuentasMensuales', es_procesada=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='antiguedadEnMeses', 
        y='cuentasMensuales',
        hue=df_temp['canceloServicio'].map({0: 'No', 1: 'Sí'}), 
        data=df_temp, 
        alpha=0.5,
        ax=ax
    )
    ax.set_title(f'Tendencia de Cancelación: {nombre_antiguedad} vs. {nombre_cuentas}')
    ax.set_xlabel(nombre_antiguedad)
    ax.set_ylabel(nombre_cuentas)
    ax.legend(title='Cancelación')
    plt.tight_layout()
    
    return fig

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
    """Genera gráfico de importancia de variables para el modelo"""
    from nombres_variables import obtener_nombre_amigable
    
    if hasattr(model, 'feature_importances_'):
        importancia = pd.DataFrame({
            'Variable': X.columns,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        # Renombrar variables a nombres amigables
        importancia['Variable'] = importancia['Variable'].apply(lambda x: obtener_nombre_amigable(x, es_procesada=True))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importancia', y='Variable', data=importancia.head(10), ax=ax)
        ax.set_title(f'Importancia de Variables - {nombre_modelo}')
        ax.set_xlabel('Importancia')
        plt.tight_layout()
        
        print(f"\nImportancia de variables - {nombre_modelo}:")
        print(importancia)
        
        return fig, importancia
        
    elif hasattr(model, 'coef_'):
        # Para Regresión Logística
        coeficientes = pd.DataFrame({
            'Variable': X.columns,
            'Coeficiente': model.coef_[0]
        }).sort_values('Coeficiente', key=abs, ascending=False)
        
        # Renombrar variables a nombres amigables
        coeficientes['Variable'] = coeficientes['Variable'].apply(lambda x: obtener_nombre_amigable(x, es_procesada=True))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coeficiente', y='Variable', data=coeficientes.head(10), ax=ax)
        ax.set_title(f'Coeficientes del Modelo - {nombre_modelo}')
        ax.set_xlabel('Coeficiente')
        plt.tight_layout()
        
        print(f"\nCoeficientes del modelo - {nombre_modelo}:")
        print(coeficientes)
        
        return fig, coeficientes

def matriz_confusion_visual(y_test, y_pred, nombre_modelo):
    """Genera una matriz de confusión visual"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Matriz de Confusión - {nombre_modelo}')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Realidad')
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['No Canceló', 'Canceló'])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['No Canceló', 'Canceló'])
    plt.tight_layout()
    
    return fig, cm

def dashboard_completo():
    """Genera resumen del dashboard sin mostrar gráficos automáticamente"""
    print("=== Generando Dashboard Completo ===\n")
    
    # Preparar datos
    X, y = preparar_datos_ml()
    
    # Solo generar resumen sin mostrar gráficos automáticamente
    print("Datos preparados para dashboard:")
    print(f"- Variables: {X.shape[1]}")
    print(f"- Registros: {X.shape[0]}")
    print(f"- Balance de clases: {y.value_counts().to_dict()}")
    
    return {
        'X': X,
        'y': y,
        'mensaje': 'Dashboard preparado. Use las funciones individuales para visualizar gráficos específicos.'
    }
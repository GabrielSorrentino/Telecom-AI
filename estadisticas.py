import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crud_service import obtener_datos, FILE_TRABAJO
from nombres_variables import renombrar_columnas_dataframe

# Constantes para etiquetas de gráficos
LABEL_CANCELADO = 'Canceló'
LABEL_NO_CANCELADO = 'No Canceló'
LABEL_CANTIDAD_CLIENTES = 'Cantidad de Clientes'
COLOR_NO_CANCELADO = '#1f77b4'
COLOR_CANCELADO = '#ff7f0e'

def obtener_dataframe():
    return pd.read_csv(FILE_TRABAJO)

def analisis_descriptivo():
    df = obtener_dataframe()
    # Renombrar columnas a nombres amigables
    df_renombrado = renombrar_columnas_dataframe(df, es_procesado=False)
    return df_renombrado.describe()

def distribucion_cancelacion():
    """Genera gráfico de distribución de cancelación"""
    df = obtener_dataframe()
    vector_cancelacion = df['canceloServicio'].map({
        True: LABEL_CANCELADO,
        False: 'No canceló'
    }).astype('string')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = vector_cancelacion.value_counts()
    counts.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Distribución de Evasión de Clientes')
    ax.set_xlabel('Cancelamiento')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    ax.set_ylim(0, counts.max() * 1.15)
    
    for i, percentage in enumerate((counts / len(vector_cancelacion)) * 100):
        ax.text(i, counts.iloc[i] + 50, f'{percentage:.1f}% ({counts.iloc[i]})', ha='center', va='bottom')
    
    plt.tight_layout()
    
    return fig, counts

def grafico_cancelacion_por_genero():
    """Genera gráfico de cancelación por género usando matplotlib"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Agrupar datos
    grouped = df.groupby(['genero', 'canceloServicio']).size().unstack()
    
    # Crear gráfico de barras agrupado
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Cancelación por Género')
    ax.set_xlabel('Género')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_contrato():
    """Genera gráfico de cancelación por tipo de contrato"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby(['tipoDeContrato', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Cancelación por Tipo de Contrato')
    ax.set_xlabel('Tipo de Contrato')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_pago():
    """Genera gráfico de cancelación por método de pago"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    grouped = df.groupby(['metodoDePago', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Cancelación por Método de Pago')
    ax.set_xlabel('Método de Pago')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_antiguedad():
    """Genera gráfico de cancelación por antigüedad"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Agrupar por rangos de antigüedad para mejor visualización
    df['rango_antiguedad'] = pd.cut(df['antiguedadEnMeses'], 
                                    bins=[0, 12, 24, 36, 48, 60, 72],
                                    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    grouped = df.groupby(['rango_antiguedad', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Cancelación por Antigüedad (meses)')
    ax.set_xlabel('Rango de Antigüedad')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_jubilado():
    """Genera gráfico de cancelación por jubilación"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df['jubilado_label'] = df['jubiladoMasDeSesenta'].map({True: 'Jubilado', False: 'No Jubilado'})
    grouped = df.groupby(['jubilado_label', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Cancelación por Jubilación')
    ax.set_xlabel('Jubilado (+60 años)')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_internet():
    """Genera gráfico de cancelación por tipo de internet"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby(['servicioDeInternet', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Cancelación por Servicio de Internet')
    ax.set_xlabel('Tipo de Servicio')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_costo_mensual():
    """Genera gráfico de cancelación por costo mensual"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Crear rangos de costo mensual
    df['rango_costo'] = pd.cut(df['cuentasMensuales'],
                               bins=[0, 30, 50, 70, 90, 120],
                               labels=['0-30', '31-50', '51-70', '71-90', '91-120'])
    grouped = df.groupby(['rango_costo', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Distribución de Cancelación por Costos Mensuales')
    ax.set_xlabel('Rango de Costo Mensual ($)')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

def grafico_cancelacion_por_cobro_total():
    """Genera gráfico de cancelación por cobro total"""
    df = obtener_dataframe()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Crear rangos de cobro total
    df['rango_cobro'] = pd.cut(df['cobroTotal'],
                               bins=[0, 500, 1000, 2000, 4000, 9000],
                               labels=['0-500', '501-1000', '1001-2000', '2001-4000', '4001-9000'])
    grouped = df.groupby(['rango_cobro', 'canceloServicio']).size().unstack()
    grouped.plot(kind='bar', ax=ax, color=[COLOR_NO_CANCELADO, COLOR_CANCELADO])
    
    ax.set_title('Distribución de Cancelación por Cobro Total')
    ax.set_xlabel('Rango de Cobro Total ($)')
    ax.set_ylabel(LABEL_CANTIDAD_CLIENTES)
    ax.legend([LABEL_NO_CANCELADO, LABEL_CANCELADO])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig

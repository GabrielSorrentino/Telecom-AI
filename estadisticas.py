import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from crud_service import obtener_datos

def obtener_dataframe():
    return pd.read_csv('Telecom_AI.csv')

def analisis_descriptivo():
    df = obtener_dataframe()
    return df.describe()

def distribucion_cancelacion():
    df = obtener_dataframe()
    vector_cancelacion = df['canceloServicio'].map({
        True: 'Canceló',
        False: 'No canceló'
    }).astype('string')
    
    plt.figure(figsize=(8, 6))
    ax = vector_cancelacion.value_counts().plot(kind='bar')
    plt.title('Distribución de Evasión de Clientes')
    plt.xlabel('Cancelamiento')
    plt.ylabel('Cantidad de Clientes')
    plt.xticks(rotation=0)
    
    plt.ylim(0, vector_cancelacion.value_counts().max() * 1.15)
    
    for i, percentage in enumerate((vector_cancelacion.value_counts() / len(vector_cancelacion)) * 100):
        ax.text(i, vector_cancelacion.value_counts().iloc[i] + 50, f'{percentage:.1f}% ({vector_cancelacion.value_counts().iloc[i]})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return vector_cancelacion.value_counts()

def grafico_cancelacion_por_genero():
    df = obtener_dataframe()
    return px.histogram(df, x='genero', text_auto=True, color='canceloServicio', barmode='group')

def grafico_cancelacion_por_contrato():
    df = obtener_dataframe()
    return px.histogram(df, x='tipoDeContrato', text_auto=True, color='canceloServicio', barmode='group')

def grafico_cancelacion_por_pago():
    df = obtener_dataframe()
    return px.histogram(df, x='metodoDePago', text_auto=True, color='canceloServicio', barmode='group')

def grafico_cancelacion_por_antiguedad():
    df = obtener_dataframe()
    return px.histogram(df, x='antiguedadEnMeses', text_auto=True, color='canceloServicio', barmode='group')

def grafico_cancelacion_por_jubilado():
    df = obtener_dataframe()
    return px.histogram(df, x='jubiladoMasDeSesenta', text_auto=True, color='canceloServicio', barmode='group')

def grafico_cancelacion_por_internet():
    df = obtener_dataframe()
    return px.histogram(df, x='servicioDeInternet', text_auto=True, color='canceloServicio', barmode='group')

def grafico_cancelacion_por_costo_mensual():
    df = obtener_dataframe()
    fig = px.histogram(df, x='cuentasMensuales', color='canceloServicio', barmode='group',
                      title='Distribución de cancelación por costos mensuales', text_auto=True)
    return fig

def grafico_cancelacion_por_cobro_total():
    df = obtener_dataframe()
    fig = px.histogram(df, x='cobroTotal', color='canceloServicio', barmode='group',
                      title='Distribución de Cancelación por Cobro Total', text_auto=True)
    return fig

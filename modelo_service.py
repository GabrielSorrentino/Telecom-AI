import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def obtener_dataframe_procesado():
    """Lee el CSV y realiza el preprocesamiento inicial para ML"""
    df = pd.read_csv('Telecom_AI.csv')
    
    # Eliminar columnas irrelevantes
    df = df.drop(columns=['ID'])
    
    # Convertir columnas booleanas a enteros
    columnas_bool = df.select_dtypes(include=['bool']).columns
    df[columnas_bool] = df[columnas_bool].astype(int)
    
    # Convertir género a binario
    df.rename(columns={'genero': 'generoMasculino'}, inplace=True)
    df['generoMasculino'] = df['generoMasculino'].map({'Masculino': 1, 'Femenino': 0}).astype(int)
    
    return df

def preprocesar_para_ml(df):
    """Aplica encoding y estandarización para ML"""
    columnas_strings = ['servicioTelefonico', 'servicioDeInternet', 'tipoDeContrato', 'metodoDePago']
    categorias_a_eliminar = ['No', 'No', 'Mes a mes', 'Cheque enviado por correo']
    columnas_escalar = ['antiguedadEnMeses', 'cuentasMensuales', 'cobroTotal']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop=categorias_a_eliminar, handle_unknown='ignore', sparse_output=False), columnas_strings),
            ('num', StandardScaler(), columnas_escalar)
        ],
        remainder='passthrough'
    )
    preprocessor.set_output(transform="pandas")
    
    df_procesado = preprocessor.fit_transform(df)
    
    # Convertir columnas cat a int
    columnas_cat = [col for col in df_procesado.columns if col.startswith('cat__')]
    df_procesado[columnas_cat] = df_procesado[columnas_cat].astype(int)
    
    # Limpiar nombres de columnas
    df_procesado.columns = df_procesado.columns.str.replace('num__', '', regex=False)
    df_procesado.columns = df_procesado.columns.str.replace('remainder__', '', regex=False)
    
    # Renombrar columnas
    df_procesado.rename(columns={
        'cat__servicioTelefonico_Múltiples líneas': 'MultiplesLineas',
        'cat__servicioTelefonico_Una línea': 'UnicaLinea',
        'cat__servicioDeInternet_DSL': 'TieneDSL',
        'cat__servicioDeInternet_Fibra óptica': 'TieneFibraOptica',
        'cat__tipoDeContrato_Dos años': 'ContratoDosAnos',
        'cat__tipoDeContrato_Un año': 'ContratoUnAno',
        'cat__metodoDePago_Cheque electrónico': 'PagoChequeElectronico',
        'cat__metodoDePago_Tarjeta de crédito (automático)': 'PagoTarjetaCredito',
        'cat__metodoDePago_Transferencia bancaria (automático)': 'PagoTransferenciaBancaria'
    }, inplace=True)
    
    return df_procesado

def eliminar_columnas_problematicas(df):
    """Elimina columnas con alta multicolinealidad"""
    columnas_a_eliminar = ['cobroTotal', 'generoMasculino', 'UnicaLinea', 'TieneDSL', 'TieneFibraOptica']
    return df.drop(columns=columnas_a_eliminar)

def preparar_datos_ml():
    """Pipeline completo de preparación de datos para ML"""
    df = obtener_dataframe_procesado()
    df_procesado = preprocesar_para_ml(df)
    df_limpio = eliminar_columnas_problematicas(df_procesado)
    
    # Separar variable objetivo
    y = df_limpio['canceloServicio']
    X = df_limpio.drop(columns=['canceloServicio'])
    
    return X, y

def dividir_datos(X, y, test_size=0.20, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def entrenar_modelo_logistico(X_train, y_train):
    """Entrena un modelo de Regresión Logística"""
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def entrenar_modelo_random_forest(X_train, y_train):
    """Entrena un modelo de Random Forest"""
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42,
        min_samples_leaf=2,
        max_features='sqrt'
    )
    model.fit(X_train, y_train)
    return model

def evaluar_modelo(model, X_test, y_test, nombre_modelo):
    """Evalúa un modelo y devuelve métricas"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n=== Resultados {nombre_modelo} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': y_pred
    }

def obtener_correlaciones(X, y):
    """Calcula y muestra la matriz de correlación"""
    df_temp = X.copy()
    df_temp['canceloServicio'] = y
    
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

def ejecutar_pipeline_completo():
    """Ejecuta todo el pipeline de ML: preparación, entrenamiento y evaluación"""
    print("=== Preparando datos ===")
    X, y = preparar_datos_ml()
    
    print("=== Mostrando correlaciones ===")
    obtener_correlaciones(X, y)
    
    print("=== Dividiendo datos ===")
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    print(f"Registros totales: {len(X)}")
    print(f"Registros para entrenar: {len(X_train)}")
    print(f"Registros para probar: {len(X_test)}")
    
    print("\n=== Entrenando modelo de Regresión Logística ===")
    model_log = entrenar_modelo_logistico(X_train, y_train)
    resultados_log = evaluar_modelo(model_log, X_test, y_test, "Regresión Logística")
    
    print("\n=== Entrenando modelo de Random Forest ===")
    model_rf = entrenar_modelo_random_forest(X_train, y_train)
    resultados_rf = evaluar_modelo(model_rf, X_test, y_test, "Random Forest")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model_log': model_log,
        'model_rf': model_rf,
        'resultados_log': resultados_log,
        'resultados_rf': resultados_rf
    }
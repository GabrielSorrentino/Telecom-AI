# Telecom-AI
Programa que simula un sistema de machine learning para determinar si clientes de una empresa de telecomunicaciones podrían abandonar el servicio.
Autor: Gabriel Sorrentino
Universidad Nacional de la Patagonia San Juan Bosco, Sede Puerto Madryn
Materia: Inteligencia Artificial

## Descripción de la Aplicación

Telecom-AI es una aplicación de escritorio completa desarrollada en Python que implementa un sistema integral de análisis de clientes de telecomunicaciones. La aplicación permite gestionar datos de clientes, realizar análisis estadísticos detallados, entrenar modelos de Machine Learning para predecir la cancelación de servicios (churn), y visualizar resultados a través de una interfaz gráfica amigable.

La aplicación trabaja sobre datos de clientes ya procesados almacenados en `Telecom_AI.csv`, que contiene información demográfica, de servicios, contratos y pagos de más de 7,000 clientes de una empresa de telecomunicaciones ficticia.

## Estructura del Repositorio

El repositorio contiene los siguientes módulos principales:

- **`crud_service.py`**: Módulo de gestión de datos que implementa operaciones CRUD (Create, Read, Update, Delete) sobre el archivo CSV de clientes.
- **`estadisticas.py`**: Módulo de análisis estadístico que genera estadísticas descriptivas, distribuciones de cancelación y gráficos categorizados por diferentes variables (género, contrato, pago, antigüedad, etc.).
- **`modelo_service.py`**: Módulo de Machine Learning que implementa el pipeline completo de preprocesamiento, entrenamiento y evaluación de dos modelos predictivos (Regresión Logística y Random Forest).
- **`visualizaciones.py`**: Módulo de visualizaciones avanzadas que genera matrices de correlación, boxplots, scatter plots y dashboards completos.
- **`main.py`**: Interfaz gráfica de usuario desarrollada con tkinter que integra todos los módulos anteriores en una aplicación amigable con pestañas organizadas.
- **`Telecom_AI.csv`**: Base de datos de clientes con información ya procesada y normalizada.
- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar la aplicación.

## Requisitos del Sistema

Para ejecutar la aplicación se necesita:

- Python 3.7 o superior
- Las librerías especificadas en `requirements.txt`:
  - pandas
  - matplotlib
  - seaborn
  - plotly
  - scikit-learn
  - numpy

## Instalación y Ejecución

1. **Clonar el repositorio** (si aún no lo has hecho):
   ```bash
   git clone https://github.com/GabrielSorrentino/Telecom-AI.git
   cd Telecom-AI
   ```

2. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**:
   ```bash
   python3 main.py
   ```

## Funcionalidades de la Aplicación

La interfaz gráfica se organiza en cuatro pestañas principales:

### 1. Gestión de Datos
- **Ver Todos los Datos**: Muestra el contenido completo del CSV en formato tabular
- **Buscar por ID**: Permite buscar un cliente específico por su identificador
- **Crear Registro**: Funcionalidad para agregar nuevos clientes al sistema
- **Actualizar Registro**: Permite modificar información de clientes existentes
- **Eliminar Registro**: Elimina clientes del sistema por ID

### 2. Estadísticas
- **Análisis Descriptivo**: Muestra estadísticas básicas (media, desviación estándar, cuartiles, etc.) de las variables numéricas
- **Distribución de Cancelación**: Gráfico de barras mostrando la proporción de clientes que cancelaron vs. los que permanecieron
- **Gráficos Categóricos**: Visualizaciones de cancelación por:
  - Género
  - Tipo de contrato
  - Método de pago
  - Antigüedad
  - Jubilación
  - Tipo de servicio de internet
- **Gráficos de Costos**: Análisis de cancelación en relación con costos mensuales y cobros totales

### 3. Modelos ML
- **Ejecutar Pipeline Completo**: Proceso automatizado que:
  - Preprocesa los datos (encoding, estandarización)
  - Elimina variables con alta multicolinealidad
  - Entrena dos modelos: Regresión Logística y Random Forest
  - Evalúa ambos modelos con métricas de performance
  - Muestra matrices de confusión y reportes de clasificación

### 4. Visualizaciones
- **Dashboard Completo**: Genera todas las visualizaciones principales en un solo proceso
- **Matriz de Correlación**: Heatmap interactivo mostrando correlaciones entre variables
- **Boxplots**: Análisis de distribución de variables numéricas vs. cancelación
- **Scatter Plots**: Diagramas de dispersión para identificar patrones en los datos
- **Importancia de Variables**: Visualización de qué factores más influyen en la cancelación

## Modelos de Machine Learning

La aplicación implementa y compara dos modelos predictivos:

### Regresión Logística
- Modelo lineal enfocado en explicabilidad
- Calcula probabilidades de cancelación
- Ideal para entender qué factores influyen en la decisión del cliente
- Alto recall (sensibilidad) para detectar clientes en riesgo

### Random Forest Classifier
- Modelo de ensamble basado en árboles de decisión
- Capaz de capturar relaciones no lineales complejas
- Mayor accuracy general
- Útil para patrones complejos en los datos

## Hallazgos Principales del Análisis

Basado en el análisis de los datos y los modelos entrenados, los principales factores que influyen en la cancelación de servicios son:

1. **Alto costo mensual**: Los clientes con planes más costosos (especialmente fibra óptica) tienen mayor probabilidad de cancelar
2. **Contratos de mes a mes**: La falta de compromiso a largo plazo facilita la salida rápida de clientes
3. **Falta de servicios de valor agregado**: La ausencia de soporte técnico y seguridad online incrementa el riesgo de cancelación
4. **Antigüedad reducida**: Los clientes nuevos (primeros meses) son los más propensos a abandonar el servicio

## Recomendaciones de Negocio

- Revisar la infraestructura técnica de la fibra óptica para mejorar la calidad del servicio
- Considerar estrategias de retención para contratos de mes a mes
- Bonificar servicios de seguridad y soporte técnico para clientes de planes premium
- Implementar programas de bienvenida específicos para nuevos clientes
- Desarrollar encuestas para entender mejor las necesidades de clientes jubilados

## Autor y Contacto

Desarrollado como proyecto académico para la materia de Inteligencia Artificial en la Universidad Nacional de la Patagonia San Juan Bosco, Sede Puerto Madryn.

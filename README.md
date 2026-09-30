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

### Para ejecución directa (con Python instalado y dependencias)

- [Git](https://git-scm.com/downloads)
- [Python 3.7+](https://www.python.org/downloads/)
- pip (incluido en instalación de Python)
- Las librerías especificadas en `requirements.txt`:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - numpy
  - Dependencias de requirements.txt
- (Recomendado) Virtual environment (venv o conda)

**Otros requisitos por Sistema Operativo:**
- **Linux**: tkinter (`sudo apt-get install python3-tk` o equivalente)
- **Windows/macOS**: tkinter incluido en instalación de Python

### Para ejecución con Docker y X11 (sin Python instalado)

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/macOS/algunas distros Linux) o [Docker CE](https://docs.docker.com/engine/install/) (Linux)
- X11

**Requisitos por sistema operativo:**
- **Linux**: X11 nativo (generalmente incluido por defecto)
- **macOS**: [XQuartz](https://www.xquartz.org/) para soporte gráfico
- **Windows**: [Xming](https://sourceforge.net/projects/xming/) o [VcXsrv](https://sourceforge.net/projects/vcxsrv/) para soporte gráfico + X11 forwarding en Docker Desktop

## Instalación y Ejecución

### Opción 1: Ejecución Directa

1. **Clonar el repositorio** (si aún no lo has hecho):
   ```bash
   git clone https://github.com/GabrielSorrentino/Telecom-AI.git
   cd Telecom-AI
   ```

2. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Instalar tkinter** (solo en Linux):
   ```bash
   sudo apt-get install python3-tk  # Debian/Ubuntu
   # o
   sudo dnf install python3-tkinter  # Fedora
   ```

4. **Ejecutar la aplicación**:
   ```bash
   python3 main.py
   ```

### Opción 2: Ejecución con Docker

Se pueden usar los scripts techmind.sh (en Linux/macOS) o techmind.ps1 (en Windows) para ejecutar la aplicación con Docker; o bien se puede construir y ejecutar el contenedor de la aplicación manualmente.

#### Ejecución en Linux/macOS

```bash
# Dar permisos de ejecución al script
chmod +x telecom.sh

# Ejecutar la aplicación
./telecom.sh
```

El script automáticamente:
- Detecta tu sistema operativo
- Configura X11 forwarding para la interfaz gráfica
- Construye la imagen desde el Dockerfile
- Ejecuta el contenedor de forma autodestruible

#### Ejecución en Windows (PowerShell)

```powershell
# Ejecutar con PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\telecom.ps1
```

**Nota importante para Windows**: Asegúrate de que Xming/VcXsrv esté ejecutándose antes de ejecutar el script, y que X11 forwarding esté habilitado en la configuración de Docker Desktop.

#### Construcción Manual de Docker

Si prefieres construir y ejecutar manualmente:

```bash
# Construir la imagen
docker build -t telecom-ai:latest .

# Ejecutar el contenedor (Linux/macOS)
# La línea `--user $(id -u):$(id -g)` es innecesaria en Windows o macOS, pero conviene en Linux para que el contenedor no dependa de root
docker run --rm \
    --user $(id -u):$(id -g) \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $(pwd):/app \
    telecom-ai:latest
```

## 🎯 Funcionalidades de la Aplicación

La interfaz se organiza en **cuatro pestañas principales**:

### 1️⃣ **Gestión de Datos**
Aquí puedes trabajar directamente con el registro de clientes:
- **Ver todos los datos**: Visualiza la tabla completa de clientes
- **Buscar por ID**: Encuentra un cliente específico rápidamente
- **Crear registro**: Agrega nuevos clientes al sistema
- **Actualizar registro**: Modifica información de clientes existentes
- **Eliminar registro**: Borra clientes del sistema

### 2️⃣ **Estadísticas**
Este es el corazón del **análisis exploratorio**. Aquí vemos **cómo se comporta la cancelación en diferentes grupos**:

#### 📊 Análisis Descriptivo
Muestra estadísticas básicas (promedio, desviación estándar, mínimo, máximo, etc.) de las variables numéricas del dataset.

#### 📈 Distribución General de Cancelación
Gráfico de barras mostrando:
- Cuántos clientes cancelaron vs. cuántos se quedaron
- Porcentaje de cancelación en la base de datos
- Ayuda a entender si hay más clientes leales o desertores

#### 🔍 Gráficos Categóricos (Qué influye en la cancelación)
Cada uno de estos gráficos te muestra **cómo cambia la cancelación según una característica específica**:

1. **Por Género**: ¿Hay diferencia entre hombres y mujeres?
2. **Por Tipo de Contrato**: ¿Los contratos de corto plazo tienen más cancelaciones?
3. **Por Método de Pago**: ¿El pago automático retiene mejor que el manual?
4. **Por Antigüedad**: ¿Los clientes nuevos se van más que los antiguos?
5. **Por Jubilación**: ¿Los jubilados se comportan diferente?
6. **Por Servicio de Internet**: ¿El tipo de internet (DSL, Fibra, etc.) importa?

#### 💰 Gráficos de Costos (El dinero importa)
- **Por Costo Mensual**: ¿Clientes que pagan más se van más?
- **Por Cobro Total**: ¿La inversión histórica del cliente predice su permanencia?

### 3️⃣ **Modelos ML** ⚡
Aquí entramos en machine learning. Este módulo **entrena y compara tres modelos inteligentes**:

#### 🤖 Los Tres Modelos:

1. **Regresión Logística** (La simple y explicable)
   - Es como un "profesor de matemáticas": muy ordenado, fácil de entender
   - Te dice exactamente cuáles factores ayudan a retener clientes
   - Es rápida y buena para empezar

2. **Random Forest** (La democracia de árboles)
   - Es como tener 80+ "árboles de decisión" votando juntos
   - Detecta relaciones más complejas que la regresión lineal
   - Mejor precisión general

3. **XGBoost** (La más avanzada)
   - Es como un estudiante aplicado que aprende de sus errores
   - Entrena árboles secuencialmente, corrigiendo errores anteriores
   - Generalmente la mejor predicción

#### 📊 Qué se evalúa de cada modelo:

El pipeline completo incluye:

**Durante la búsqueda de parámetros (Sintonía):**
- Prueba cientos de combinaciones de parámetros
- Usa validación cruzada de 5-fold para asegurar que funcione bien en datos nuevos
- Elige la combinación que maximiza el F1 (balance entre precisión y sensibilidad)

**Evaluación Final (en los datos de prueba):**
- **Exactitud (Accuracy)**: De cada 100 predicciones, ¿cuántas aciertan?
- **Precisión**: Cuando dice "este cliente se va", ¿qué tan seguido tiene razón?
- **Sensibilidad (Recall)**: De todos los clientes que realmente se fueron, ¿cuántos logró identificar? (crucial para negocios)
- **Especificidad**: De los clientes que se quedaron, ¿cuántos identificó correctamente?
- **F1**: Balance perfecto entre precisión y sensibilidad
- **ROC-AUC**: Medida general de calidad del modelo

**Matriz de Confusión:**
Muestra cuatro categorías:
- **Verdaderos Positivos (VP)**: Predijo cancelación y canceló ✓
- **Falsos Positivos (FP)**: Predijo cancelación pero se quedó ✗
- **Falsos Negativos (FN)**: No predijo cancelación pero se fue ✗ (lo peor)
- **Verdaderos Negativos (VN)**: Predijo que se queda y se quedó ✓

### 4️⃣ **Visualizaciones** 🎨
Gráficos avanzados para entender los datos en profundidad:

- **Matriz de Correlación**: Heatmap mostrando qué variables están relacionadas entre sí y cuáles más influyen en la cancelación
- **Boxplots**: Compara la distribución de variables numéricas entre clientes que cancelaron vs. los que se quedaron
- **Scatter Plots**: Diagrama de dispersión mostrando la relación entre antigüedad y costo mensual, coloreado por cancelación
- **Importancia de Variables**: Barra de cuáles son los factores más determinantes para cada modelo

---

## 🧠 Modelos de Machine Learning Explicados

### **Regresión Logística**
- **¿Para qué?** Predicción rápida y explicable
- **¿Cuándo usarla?** Cuando necesitas entender el "por qué"
- **Ventaja:** Cada coeficiente te dice cuánto influye cada factor
- **Limitación:** Solo captura relaciones lineales

### **Random Forest**
- **¿Para qué?** Predicción con buena precisión general
- **¿Cuándo usarla?** Cuando necesitas equilibrio entre precisión y velocidad
- **Ventaja:** Detecta relaciones complejas y no lineales
- **Limitación:** Menos interpretable que regresión logística

### **XGBoost**
- **¿Para qué?** Máxima precisión predictiva
- **¿Cuándo usarla?** Cuando necesitas el mejor rendimiento posible
- **Ventaja:** Generalmente el más preciso de los tres
- **Limitación:** Más lento que los otros y más "caja negra"

---

## 💡 Hallazgos Principales del Análisis

Basado en el análisis de los datos y los modelos entrenados, los principales factores que influyen en la cancelación son:

1. **Alto costo mensual** ⚠️
   - Clientes con planes más costosos (especialmente fibra óptica) tienen mayor probabilidad de cancelar
   - Posible causa: precio-calidad percibida

2. **Contratos de mes a mes** 📅
   - La falta de compromiso a largo plazo facilita la salida rápida
   - Estos clientes son menos "pegajosos"

3. **Falta de servicios de valor agregado** 🔧
   - Ausencia de soporte técnico y seguridad online incrementa el riesgo
   - Los extras retienen clientes

4. **Antigüedad reducida** 🆕
   - Clientes nuevos (primeros meses) son los más propensos a abandonar
   - Los primeros 3 meses son críticos

5. **Jubilados sin opciones** 👴
   - Clientes mayores jubilados tienen patrones especiales de comportamiento
   - Necesitan atención específica

---

## 🎯 Recomendaciones de Negocio

📌 **Para retener clientes:**

1. **Revisar la infraestructura de fibra óptica**
   - Si el servicio es caro pero la calidad es mala, arreglarlo es urgente
   - Considera revisiones de velocidad/estabilidad

2. **Estrategias especiales para contratos de mes a mes**
   - Incentiva upgrading a contratos de 1 o 2 años
   - Ofrece descuentos por permanencia más larga

3. **Empaquetar servicios de seguridad**
   - Bonifica soporte técnico y seguridad online para clientes de planes premium
   - Aumenta el valor percibido del servicio

4. **Programa de bienvenida agresivo**
   - Implementa actividades especiales para nuevos clientes (primeros 3 meses)
   - Contacto proactivo, ofertas especiales, etc.

5. **Encuestas dirigidas a jubilados**
   - Entiende mejor sus necesidades específicas
   - Desarrolla paquetes personalizados para este segmento

---

## 📦 Dependencias

```
pandas          # Manipulación y análisis de datos
matplotlib      # Visualización gráfica
seaborn         # Gráficos estadísticos mejorados
scikit-learn    # Modelos de ML y métricas
xgboost         # Modelo XGBoost
numpy           # Operaciones numéricas
tkinter         # Interfaz gráfica
```

Instálalas todas con:
```bash
pip install -r requirements.txt
```

---

## 📄 Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Punto de entrada - ejecuta la aplicación gráfica |
| `crud_service.py` | Operaciones CRUD sobre el CSV |
| `estadisticas.py` | Gráficos estadísticos y descriptivos |
| `modelo_service.py` | Pipeline completo de ML (el más importante) |
| `visualizaciones.py` | Gráficos avanzados y matrices |
| `nombres_variables.py` | Mapeo de nombres técnicos a nombres amigables |
| `Telecom_AI.csv` | Base de datos de clientes |
| `requirements.txt` | Dependencias del proyecto |
| `Dockerfile` | Configuración para ejecutar con Docker |
| `telecom.sh` / `telecom.ps1` | Scripts para ejecutar con Docker |

---

## 🔄 Flujo de Trabajo Típico

1. **Exploración**: Abre la pestaña "Estadísticas" para entender los datos
2. **Análisis**: Revisa "Visualizaciones" para ver relaciones complejas
3. **Predicción**: Ejecuta "Modelos ML" para entrenar los tres modelos
4. **Decisión**: Compara resultados y elige el modelo más confiable
5. **Acción**: Usa las recomendaciones para implementar estrategias de retención

---

## 👨‍💻 Autor

Desarrollado como proyecto académico para la materia **Inteligencia Artificial** en la:

**Universidad Nacional de la Patagonia San Juan Bosco**  
Sede Puerto Madryn  
Patagonia, Argentina 🇦🇷

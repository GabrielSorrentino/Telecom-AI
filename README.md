# Telecom-AI
Programa que simula un sistema de machine learning para determinar si clientes de una empresa de telecomunicaciones podrían abandonar el servicio.
Autor: Gabriel Sorrentino
Universidad Nacional de la Patagonia San Juan Bosco, Sede Puerto Madryn
Materia: Inteligencia Artificial

# **PENDIENTE**
Acomodar el presente README según cómo quedará estructurado finalmente el repositorio, en base a lo que dice este programa y a lo que se evalúa en los comentarios de logica.py.

## Introducción
En este desafío, se simuló una situación en la que se pidió analizar el motivo de la gran tasa de cancelaciones (alrededor del 25% del total de clientes, como se puede explorar en el notebook) que sufre la empresa ficticia Telecom X LATAM, encargada de proporcionar servicios de red y telecomunicaciones. Por lo cual, tuve que procesar los datos de un JSON, normalizarlos y modificarlos en varios aspectos para que se puedan entender y manipular fácilmente, y realizar observaciones probabilísticas con los mismos.

## El repositorio
El repositorio consiste en el presente Markdown, un JSON que contiene los datos con los que trabajé, y un Notebook .ipynb, en el que ejecuté el programa con Google Colab. El JSON contiene los datos de varios clientes, como el ID, y la información de si cancelaron o no el servicio de Telecom X ('Churn'), y otros atributos que consisten en otros JSON anidados, que reflejan los servicios telefónicos y de internet, su género y su antigüedad como clientes de la empresa, y datos referidos al pago del servicio según el tipo de contrato y los servicios optados, entre otra información. El programa no lee el .json directamente del repositorio ni de la carpeta donde se lo clone, sino de una URL específica; el JSON está en el repositorio a modo de referencia.

## El programa

El programa primeramente importa el JSON, lee y muestra las columnas iniciales de dicha base de datos sin normalizar, y luego crea un DataFrame adecuadamente normalizado a partir del JSON (la normalización, por suerte, no incrementó la cantidad de filas por lo que no generó problemas como repetición de IDs). Luego, muestra que los clientes que no está indicado si cancelaron o no el servicio (columna 'Churn' vacía) o que no tienen un costo total señalizado (debería ser 0 o 0.0) son escasos, y a continuación, limpia el DataFrame: elimina las filas con Churn vacío, unifica la información del servicio telefónico (si se cuenta o no con éste, y si se cuenta con múltiples líneas o línea simple) en una sola columna, convierte gran parte de las columnas (que eran columnas de 'object' pero consistentes en strings) en columnas de strings, convierte la mayor parte de las columnas (que son columnas de enteros que valen 0 o 1, o, principalmente, de 'objects' consistentes en strings que valen 'Yes' o 'No', o bien 'No internet service', que para nuestro análisis equivale a 'No') en columnas de booleanos (más fáciles y eficientes de procesar), y convierte la columna de pagos totales en una de números decimales sin NaNs (que fueron reemplazados por ceros). Luego, se traduce al español los nombres de las columnas y los nombres de las casillas que estaban en inglés, para que quedaran más legibles, y se crea una nueva columna de cuotas diarias.

Luego, viene el análisis estadístico. Se muestran la media, desviación estándar, cuartiles, mínimos y máximos de las columnas numéricas del DataFrame, se muestran las proporciones y cantidades de clientes que abandonaron y continuaron con el servicio de Telecom X, y se muestra en distintos diagramas de barras la categorización de los clientes que abandonaron y continuaron con dicho servicio con respecto al género, la edad (si son o no jubilados), el tipo de servicio de internet, el método de pago, y el tipo de contrato, entre otros factores. Por último, un no muy extenso informe detalla las observaciones realizadas, incluyendo la evidente variabilidad de las antigüedades de los clientes en meses y de los costos pagados, así como los principales responsables de la fuga masiva de clientes; y da recomendaciones de qué debería hacer la empresa para enfrentar la gran tasa de cancelaciones.

## Ejecución del programa

Para ejecutar el programa, primeramente hay que contar con una cuenta de Gmail, insertar el Notebook a Google Drive, y luego abrirlo; se abrirá con Google Colab. A continuación, hay que presionar el botón 'Ejecutar todas' para que se ejecute todo el código secuencialment y se visualicen los resultados del proceso de normalización y limpieza del DataFrame, y los gráficos. El informe se encuentra abajo de todo.

# **ADVERTENCIA**

Ahora arranca el README de la parte 2 del programa

## Introducción

Tras los resultados del análisis hecho en el challenge anterior, la empresa Telecom X LATAM me incluyó en el equipo de Machine Learning, en el que debí procesar y hacer un conjunto de análisis con los datos del challenge anterior, eliminar filas y pasarlos a formato numérico para crear dos modelos de ML y hacer que éstos estudien la mayor parte de los datos en cuestión para que, con el resto de los datos, se pueda testear la confiabilidad de dichos modelos, los resultados de los análisis, y por qué tomaron las decisiones que tomaron a la hora de decidir si un cliente supuestamente iba a irse o no.

## El repositorio

El repositorio consiste en tres archivos:
* El presente `README.md`.
* Un archivo en formato `CSV` en el que están almacenados los datos con los que trabaja el notebook.
* El programa, en formato `.ipynb`, que se ejecuta mediante Google Colab.

## Ejecución del programa

Para que el programa funcione, hay que contar con una cuenta de Gmail y acceso a Google Colab. Se debe abrir el Notebook abriendo el archivo `.ipynb` (una forma de abrirlo es contar con Google Drive, arrastrar el Notebook allí o a alguna carpeta del mismo —recomiendo que sea en la carpeta de nombre *Colab Notebooks*, ya que allí es donde se guardan por defecto los notebooks de Colab—, hacer clic derecho sobre el archivo, seleccionar *Abrir con*, y elegir *Google Colaboratory*). 

El archivo `datos_tratados.csv` (véase este mismo repositorio) debe cargarse en la carpeta de *Archivos*, al costado izquierdo de la pantalla de la página de Google Colab, a la misma altura que la carpeta `sample_data` (NO dentro de ella); es decir, en la ruta `/content/datos_tratados.csv` dentro de la máquina virtual Linux generada en el notebook de Colab.

### Librerías utilizadas
Al principio del código, se importarán las librerías necesarias para el proyecto:
* **Pandas** para la manipulación y análisis del DataFrame.
* **Warnings** para desactivar las advertencias que aparezcan como consecuencia de la ejecución del programa.
* **Matplotlib** (y Pyplot más en particular) y **Seaborn** para el procesamiento de datos e impresión de gráficos.
* **Sci-Kit Learn** para el encoding y transformaciones de las columnas (a valores numéricos), estandarizar los datos, dividir el DataFrame en conjuntos de entrenamiento y prueba, generar los modelos, entrenarlos, testearlos, y obtener las métricas de evaluación.

## El programa y Análisis de Datos

El programa trabaja con el mismo DataFrame del challenge anterior (Telecom X LATAM - Parte 1), con los datos ya normalizados y las columnas en español. Lo primero que hace es importar el CSV y realizar una limpieza profunda:
1. Se eliminan variables ruidosas y redundantes (como el ID del cliente y las cuentas diarias).
2. Se analizan y transforman las variables categóricas ('object') a formato numérico utilizando técnicas de mapeo y *One-Hot Encoding* (variables dummy).
3. Se realiza un análisis de **multicolinealidad** mediante mapas de calor (correlación de Pearson) y Factor de Inflación de la Varianza (VIF), decidiendo eliminar variables altamente correlacionadas entre sí (como el *Cobro total*, y redundancias en los tipos de servicio de internet y telefonía) para garantizar la estabilidad matemática de los modelos.

## Modelado de Machine Learning

Una vez estructurado el conjunto de datos limpio (`X`) y aislada la variable objetivo (`y` - Cancelación del servicio), se dividieron los registros utilizando un split de 80% para entrenamiento y 20% para pruebas, aplicando estratificación para mantener la proporción de las clases.

Se entrenaron y compararon dos modelos predictivos:
* **Regresión Logística:** Un modelo lineal enfocado en la explicabilidad y el cálculo de probabilidades.
* **Random Forest Classifier:** un modelo de ensamble basado en árboles de decisión capaz de capturar relaciones no lineales complejas.

## Resultados y Conclusiones

Tras evaluar ambos modelos mediante Matrices de Confusión y Reportes de Clasificación, se determinó que **la Regresión Logística es el modelo óptimo para este problema de negocio.** Aunque el Random Forest obtuvo una exactitud (*accuracy*) marginalmente superior, la Regresión Logística alcanzó un **Recall (Sensibilidad) del 79%** a la hora de identificar la clase positiva (clientes que cancelan), frente a un deficiente 46% del Random Forest. En este contexto, es preferible lidiar con falsos positivos que dejar escapar a clientes reales a punto de darse de baja.

**Hallazgos de Negocio (Feature Importance):**
El análisis de los coeficientes reveló que la fuga de clientes no es aleatoria. Los principales factores ("drivers") que impulsan las cancelaciones son:
1. El **alto costo mensual** de los servicios, fuertemente vinculado a la tenencia de Fibra Óptica.
2. Los **contratos de mes a mes**, que facilitan la salida rápida del cliente.
3. La carencia de servicios de valor agregado, específicamente la **falta de Soporte Técnico y Seguridad Online**. 

Se recomienda a la empresa revisar la infraestructura técnica de la fibra óptica, y considerar estrategias de retención enfocadas en bonificar servicios de seguridad y soporte para atar el valor percibido a las facturas de alto costo.

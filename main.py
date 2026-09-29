import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from crud_service import (
    inicializar_csv_trabajo, obtener_datos, crear_registro, 
    leer_registro, actualizar_registro, eliminar_registro
)
from estadisticas import (
    analisis_descriptivo, distribucion_cancelacion,
    grafico_cancelacion_por_genero, grafico_cancelacion_por_contrato,
    grafico_cancelacion_por_pago, grafico_cancelacion_por_antiguedad,
    grafico_cancelacion_por_jubilado, grafico_cancelacion_por_internet,
    grafico_cancelacion_por_costo_mensual, grafico_cancelacion_por_cobro_total
)
from modelo_service import ejecutar_pipeline_completo
import threading

# Constantes para mensajes
INFO_TITLE = "Información"

# Constantes para modelos
MODELO_REGRESION_LOGISTICA = "Regresión Logística"
MODELO_RANDOM_FOREST = "Random Forest"

# Constantes para etiquetas de gráficos
LABEL_CANCELADO = "Canceló"
LABEL_NO_CANCELADO = "No Canceló"

# Constantes para títulos de ventanas
TITULO_MATRIZ_CORRELACION = "Matriz de Correlación"

class TelecomAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Telecom-AI - Sistema de Análisis de Clientes")
        self.root.geometry("1200x800")
        
        # Variable para almacenar resultados de modelos
        self.resultados_modelos = None
        
        # Manejar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Crear el CSV de trabajo si no existe
        inicializar_csv_trabajo()
        
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear las diferentes pestañas
        self.crear_pestana_gestion_datos()
        self.crear_pestana_estadisticas()
        self.crear_pestana_modelos()
    
    def on_closing(self):
        """Maneja el cierre de la aplicación de forma inmediata"""
        # 1. Cerrar todas las ventanas hijas (top level) con manejo robusto
        for widget in list(self.root.winfo_children()):
            if isinstance(widget, tk.Toplevel):
                try:
                    widget.destroy()
                except tk.TclError:
                    pass  # Ventana ya cerrada, ignorar error
                except Exception as e:
                    print(f"Error cerrando ventana hija: {e}")
        
        # 2. Cerrar todas las figuras de matplotlib abiertas
        try:
            plt.close('all')
        except:
            pass
        
        # 3. Destruir la ventana principal inmediatamente
        self.root.destroy()
    
    def _run_thread_safe(self, target_func):
        """Ejecuta una función en un hilo daemon (se termina automáticamente al cerrar la app)"""
        thread = threading.Thread(target=target_func, daemon=True)
        thread.start()
        
    def crear_pestana_gestion_datos(self):
        """Crea la pestaña de gestión de datos (CRUD)"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Gestión de Datos")
        
        # Frame principal
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Ver Todos los Datos", command=self.ver_todos_datos).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Buscar por ID", command=self.buscar_por_id).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Crear Registro", command=self.crear_registro_gui).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Actualizar Registro", command=self.actualizar_registro_gui).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Eliminar Registro", command=self.eliminar_registro_gui).pack(side='left', padx=5)
        
        # Mapeo de nombres de columnas a nombres amigables
        self.nombres_columnas = {
            'ID': 'ID',
            'canceloServicio': 'Canceló el servicio',
            'genero': 'Género',
            'jubiladoMasDeSesenta': 'Jubilado (+60)',
            'conPareja': 'Con pareja',
            'conDependientes': 'Con dependientes',
            'antiguedadEnMeses': 'Antigüedad (meses)',
            'servicioTelefonico': 'Servicio telefónico',
            'servicioDeInternet': 'Servicio de internet',
            'seguridadEnLinea': 'Seguridad en línea',
            'copiaDeSeguridadEnLinea': 'Copia de seguridad',
            'proteccionDeDispositivos': 'Protección de dispositivos',
            'soporteTecnico': 'Soporte técnico',
            'streamingTV': 'Streaming TV',
            'streamingDePeliculas': 'Streaming películas',
            'tipoDeContrato': 'Tipo de contrato',
            'tieneFacturaElectronica': 'Factura electrónica',
            'metodoDePago': 'Método de pago',
            'cuentasMensuales': 'Cuentas mensuales',
            'cobroTotal': 'Cobro total'
        }
        
        # Crear Treeview para mostrar datos en formato tabla
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill='both', expand=True, pady=5)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Treeview
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.tree.pack(side='left', fill='both', expand=True)
        
        # Configurar columnas iniciales (vacías)
        columnas_originales = list(self.nombres_columnas.keys())
        
        self.tree['columns'] = columnas_originales
        self.tree.column('#0', width=0, stretch='no')  # Ocultar columna #0
        
        for col in columnas_originales:
            self.tree.column(col, width=100, anchor='center')
            self.tree.heading(col, text=self.nombres_columnas[col], anchor='center')
        
    def crear_pestana_estadisticas(self):
        """Crea la pestaña de estadísticas"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Estadísticas")
        
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Botones de estadísticas
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Análisis Descriptivo", command=self.mostrar_analisis_descriptivo).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Distribución de Cancelación", command=self.mostrar_distribucion_cancelacion).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Gráfico por Género", command=self.mostrar_grafico_genero).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Gráfico por Contrato", command=self.mostrar_grafico_contrato).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Gráfico por Pago", command=self.mostrar_grafico_pago).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Gráfico por Antigüedad", command=self.mostrar_grafico_antiguedad).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Matriz de Correlación", command=self.mostrar_matriz_correlacion).pack(side='left', padx=5)
        
        # Área de texto para resultados
        self.stats_text = tk.Text(main_frame, wrap='word', height=20)
        stats_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
        
    def crear_pestana_modelos(self):
        """Crea la pestaña de modelos de Machine Learning"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Modelos ML")
        
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Botones de modelos
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Ejecutar Pipeline Completo", command=self.ejecutar_pipeline_thread).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Ver Visualización de Modelos", command=self.ver_visualizacion_modelos).pack(side='left', padx=5)
        
        # Área de texto para resultados de modelos
        self.model_text = tk.Text(main_frame, wrap='word', height=20)
        model_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.model_text.yview)
        self.model_text.configure(yscrollcommand=model_scrollbar.set)
        
        self.model_text.pack(side='left', fill='both', expand=True)
        model_scrollbar.pack(side='right', fill='y')
        
    # Métodos de gestión de datos
    def _formatear_valor(self, valor):
        """Formatea un valor para mostrar en la tabla"""
        if isinstance(valor, bool):
            return 'Sí' if valor else 'No'
        return str(valor)
    
    def _registro_a_valores(self, registro):
        """Convierte un registro a lista de valores formateados"""
        valores = []
        for col in self.nombres_columnas.keys():
            valor = registro.get(col, '')
            valores.append(self._formatear_valor(valor))
        return valores
    
    def _limpiar_treeview(self):
        """Limpia todos los elementos del treeview"""
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def ver_todos_datos(self):
        """Muestra todos los datos del CSV en formato tabla"""
        try:
            datos = obtener_datos()
            self._limpiar_treeview()
            
            if datos:
                for registro in datos:
                    valores = self._registro_a_valores(registro)
                    self.tree.insert('', 'end', values=valores)
                
                messagebox.showinfo(INFO_TITLE, f"Se cargaron {len(datos)} registros")
            else:
                messagebox.showwarning("Advertencia", "No hay datos en el archivo.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer datos: {str(e)}")
    
    def _mostrar_registro_en_tabla(self, registro):
        """Muestra un solo registro en la tabla"""
        self._limpiar_treeview()
        valores = self._registro_a_valores(registro)
        self.tree.insert('', 'end', values=valores)
    
    def buscar_por_id(self):
        """Busca un registro por ID y lo resalta en la tabla"""
        id_buscar = simpledialog.askstring("Buscar", "Ingrese el ID del cliente:")
        if not id_buscar:
            return
        
        try:
            registro = leer_registro(id_buscar)
            
            if registro:
                self._mostrar_registro_en_tabla(registro)
                messagebox.showinfo("Encontrado", f"Registro con ID {id_buscar} encontrado")
            else:
                messagebox.showwarning("No encontrado", f"No se encontró registro con ID: {id_buscar}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al buscar: {str(e)}")
    
    def crear_registro_gui(self):
        """Interfaz para crear un nuevo registro"""
        messagebox.showinfo(INFO_TITLE, "Función de crear registro - Para crear un registro manualmente, editar el CSV directamente")
    
    def actualizar_registro_gui(self):
        """Interfaz para actualizar un registro"""
        id_actualizar = simpledialog.askstring("Actualizar", "Ingrese el ID del cliente a actualizar:")
        if id_actualizar:
            messagebox.showinfo(INFO_TITLE, f"Función de actualizar registro para ID: {id_actualizar}\nImplementación pendiente - editar CSV directamente")
    
    def eliminar_registro_gui(self):
        """Interfaz para eliminar un registro"""
        id_eliminar = simpledialog.askstring("Eliminar", "Ingrese el ID del cliente a eliminar:")
        if id_eliminar:
            try:
                if eliminar_registro(id_eliminar):
                    messagebox.showinfo("Éxito", f"Registro con ID {id_eliminar} eliminado correctamente")
                    self.ver_todos_datos()
                else:
                    messagebox.showwarning("Advertencia", f"No se encontró registro con ID: {id_eliminar}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al eliminar: {str(e)}")
    
    # Métodos de estadísticas
    def _mostrar_grafico_en_ventana(self, fig, titulo):
        """Muestra un gráfico de matplotlib en una ventana tkinter"""
        ventana = tk.Toplevel(self.root)
        ventana.title(titulo)
        ventana.geometry("800x600")
        
        canvas = FigureCanvasTkAgg(fig, master=ventana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def mostrar_analisis_descriptivo(self):
        """Muestra el análisis descriptivo"""
        try:
            stats = analisis_descriptivo()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, str(stats))
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis descriptivo: {str(e)}")
    
    def mostrar_distribucion_cancelacion(self):
        """Muestra la distribución de cancelación"""
        try:
            fig, counts = distribucion_cancelacion()
            self._mostrar_grafico_en_ventana(fig, "Distribución de Cancelación")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Distribución de Cancelación:\n")
            self.stats_text.insert(tk.END, str(counts))
        except Exception as e:
            messagebox.showerror("Error", f"Error en distribución: {str(e)}")
    
    def mostrar_grafico_genero(self):
        """Muestra gráfico por género"""
        try:
            fig = grafico_cancelacion_por_genero()
            self._mostrar_grafico_en_ventana(fig, "Cancelación por Género")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por género generado en ventana de la aplicación")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    def mostrar_grafico_contrato(self):
        """Muestra gráfico por contrato"""
        try:
            fig = grafico_cancelacion_por_contrato()
            self._mostrar_grafico_en_ventana(fig, "Cancelación por Tipo de Contrato")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por contrato generado en ventana de la aplicación")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    def mostrar_grafico_pago(self):
        """Muestra gráfico por método de pago"""
        try:
            fig = grafico_cancelacion_por_pago()
            self._mostrar_grafico_en_ventana(fig, "Cancelación por Método de Pago")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por método de pago generado en ventana de la aplicación")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    def mostrar_grafico_antiguedad(self):
        """Muestra gráfico por antigüedad"""
        try:
            fig = grafico_cancelacion_por_antiguedad()
            self._mostrar_grafico_en_ventana(fig, "Cancelación por Antigüedad")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por antigüedad generado en ventana de la aplicación")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    # Métodos de modelos
    def ejecutar_pipeline_thread(self):
        """Ejecuta el pipeline de ML en un hilo separado"""
        self.model_text.delete(1.0, tk.END)
        self.model_text.insert(tk.END, "Ejecutando pipeline de Machine Learning...\n")
        self.model_text.insert(tk.END, "Esto puede tomar varios segundos...\n")
        
        self._run_thread_safe(self.ejecutar_pipeline)
    
    def ejecutar_pipeline(self):
        """Ejecuta el pipeline completo de ML"""
        try:
            resultados = ejecutar_pipeline_completo()
            
            # Guardar resultados para visualización
            self.resultados_modelos = resultados
            
            self.model_text.delete(1.0, tk.END)
            self.model_text.insert(tk.END, "=== Pipeline de Machine Learning Completado ===\n\n")
            self.model_text.insert(tk.END, f"Accuracy Regresión Logística: {resultados['resultados_log']['accuracy']:.4f}\n")
            self.model_text.insert(tk.END, f"Accuracy Random Forest: {resultados['resultados_rf']['accuracy']:.4f}\n\n")
            self.model_text.insert(tk.END, "=== Reporte Regresión Logística ===\n")
            self.model_text.insert(tk.END, resultados['resultados_log']['classification_report'])
            self.model_text.insert(tk.END, "\n=== Reporte Random Forest ===\n")
            self.model_text.insert(tk.END, resultados['resultados_rf']['classification_report'])
            
            messagebox.showinfo("Éxito", "Pipeline de ML completado exitosamente")
        except Exception as e:
            self.model_text.insert(tk.END, f"\nError: {str(e)}")
            messagebox.showerror("Error", f"Error en pipeline: {str(e)}")
    
    def ver_visualizacion_modelos(self):
        """Abre ventana de visualización de resultados de modelos"""
        if self.resultados_modelos is None:
            messagebox.showwarning("Modelos no evaluados", "Aún no se han generado y evaluado los modelos.\nPor favor ejecute el pipeline completo primero.")
            return
        
        self._crear_ventana_visualizacion_modelos()
    
    def _crear_ventana_visualizacion_modelos(self):
        """Crea ventana con visualizaciones gráficas de los modelos"""
        ventana = tk.Toplevel(self.root)
        ventana.title("Visualización de Modelos ML")
        ventana.geometry("1000x700")
        
        # Manejar cierre de esta ventana hija
        ventana.protocol("WM_DELETE_WINDOW", lambda: self._cerrar_ventana_segura(ventana))
        
        # Notebook para pestañas de visualización
        notebook = ttk.Notebook(ventana)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestaña 1: Comparación de Accuracy
        self._crear_pestana_comparacion_accuracy(notebook)
        
        # Pestaña 2: Matrices de Confusión
        self._crear_pestana_matrices_confusion(notebook)
        
        # Pestaña 3: Métricas Detalladas
        self._crear_pestana_metricas_detalladas(notebook)
    
    def _cerrar_ventana_segura(self, ventana):
        """Cierra una ventana de forma segura limpiando recursos"""
        try:
            ventana.destroy()
        except tk.TclError:
            pass
    
    def _crear_pestana_comparacion_accuracy(self, notebook):
        """Crea pestaña con comparación de accuracy entre modelos"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Comparación Accuracy")
        
        resultados = self.resultados_modelos
        accuracy_log = resultados['resultados_log']['accuracy']
        accuracy_rf = resultados['resultados_rf']['accuracy']
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(8, 6))
        modelos = [MODELO_REGRESION_LOGISTICA, MODELO_RANDOM_FOREST]
        accuracies = [accuracy_log, accuracy_rf]
        colores = ['#1f77b4', '#ff7f0e']
        
        bars = ax.bar(modelos, accuracies, color=colores)
        ax.set_title('Comparación de Accuracy entre Modelos')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Agregar etiquetas con valores
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Integrar con tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _crear_pestana_matrices_confusion(self, notebook):
        """Crea pestaña con matrices de confusión"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Matrices de Confusión")
        
        resultados = self.resultados_modelos
        
        # Crear subplots para las dos matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matriz de confusión Regresión Logística
        cm_log = resultados['resultados_log']['confusion_matrix']
        sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title(MODELO_REGRESION_LOGISTICA)
        ax1.set_xlabel('Predicción')
        ax1.set_ylabel('Realidad')
        ax1.set_xticks([0.5, 1.5])
        ax1.set_xticklabels([LABEL_NO_CANCELADO, LABEL_CANCELADO])
        ax1.set_yticks([0.5, 1.5])
        ax1.set_yticklabels([LABEL_NO_CANCELADO, LABEL_CANCELADO])
        
        # Matriz de confusión Random Forest
        cm_rf = resultados['resultados_rf']['confusion_matrix']
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
        ax2.set_title('Random Forest')
        ax2.set_xlabel('Predicción')
        ax2.set_ylabel('Realidad')
        ax2.set_xticks([0.5, 1.5])
        ax2.set_xticklabels([LABEL_NO_CANCELADO, LABEL_CANCELADO])
        ax2.set_yticks([0.5, 1.5])
        ax2.set_yticklabels([LABEL_NO_CANCELADO, LABEL_CANCELADO])
        
        plt.tight_layout()
        
        # Integrar con tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _crear_pestana_metricas_detalladas(self, notebook):
        """Crea pestaña con métricas detalladas"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Métricas Detalladas")
        
        resultados = self.resultados_modelos
        
        # Extraer métricas del reporte de clasificación
        # Parsear el reporte para obtener precision, recall, f1-score
        from sklearn.metrics import precision_recall_fscore_support
        
        y_test = resultados['y_test']
        pred_log = resultados['resultados_log']['predictions']
        pred_rf = resultados['resultados_rf']['predictions']
        
        # Calcular métricas para cada modelo
        precision_log, recall_log, f1_log, _ = precision_recall_fscore_support(y_test, pred_log, average='weighted')
        precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, pred_rf, average='weighted')
        
        # Crear gráfico comparativo de métricas
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metricas = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metricas))
        width = 0.35
        
        metricas_log = [precision_log, recall_log, f1_log]
        metricas_rf = [precision_rf, recall_rf, f1_rf]
        
        bars1 = ax.bar(x - width/2, metricas_log, width, label=MODELO_REGRESION_LOGISTICA, color='#1f77b4')
        bars2 = ax.bar(x + width/2, metricas_rf, width, label='Random Forest', color='#ff7f0e')
        
        ax.set_title('Comparación de Métricas por Modelo')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metricas)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Agregar etiquetas con valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Integrar con tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    # Métodos de visualizaciones (integrados en Estadísticas)
    
    def mostrar_matriz_correlacion(self):
        """Muestra la matriz de correlación"""
        try:
            from modelo_service import preparar_datos_ml
            from visualizaciones import mostrar_matriz_correlacion
            
            X, y = preparar_datos_ml()
            fig, _ = mostrar_matriz_correlacion(X, y)
            self._mostrar_grafico_en_ventana(fig, TITULO_MATRIZ_CORRELACION)
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Matriz de correlación generada en ventana de la aplicación")
        except Exception as e:
            messagebox.showerror("Error", f"Error en matriz de correlación: {str(e)}")

def main():
    root = tk.Tk()
    TelecomAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

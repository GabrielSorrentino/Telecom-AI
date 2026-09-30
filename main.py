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
from modelo_service import ejecutar_pipeline_completo, formatear_reporte_pipeline, N_FOLDS
from nombres_variables import obtener_nombre_amigable
import threading

# Constantes para mensajes
INFO_TITLE = "Información"
SUCCESS_TITLE = "Éxito"

# Constantes para métricas
SENSIBILIDAD_RECALL = "Sensibilidad (Recall)"

# Constantes para modelos
MODELO_REGRESION_LOGISTICA = "Regresión Logística"
MODELO_RANDOM_FOREST = "Random Forest"
MODELO_XGBOOST = "XGBoost"

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
        for widget in self.root.winfo_children():
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
        except Exception:
            pass  # Ignorar errores al cerrar figuras matplotlib
        
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
    def _formatear_valor(self, valor, columna):
        """Formatea un valor para mostrar en la tabla"""
        if columna in ['cuentasMensuales', 'cobroTotal']:
            try:
                valor_float = float(valor)
                return f'${valor_float:.2f}'
            except (ValueError, TypeError):
                return str(valor)
        if isinstance(valor, bool):
            return 'Sí' if valor else 'No'
        if valor in ['True', 'False']:
            return 'Sí' if valor == 'True' else 'No'
        return str(valor)
    
    def _registro_a_valores(self, registro):
        """Convierte un registro a lista de valores formateados"""
        valores = []
        for col in self.nombres_columnas.keys():
            valor = registro.get(col, '')
            valores.append(self._formatear_valor(valor, col))
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
        self._ventana_registro(modo='crear')
    
    def actualizar_registro_gui(self):
        """Interfaz para actualizar un registro"""
        id_actualizar = simpledialog.askstring("Actualizar", "Ingrese el ID del cliente a actualizar:")
        if not id_actualizar:
            return
        
        registro = leer_registro(id_actualizar)
        if not registro:
            messagebox.showwarning("No encontrado", f"No se encontró registro con ID: {id_actualizar}")
            return
        
        self._ventana_registro(modo='actualizar', registro_existente=registro, id_registro=id_actualizar)
    
    def eliminar_registro_gui(self):
        """Interfaz para eliminar un registro"""
        id_eliminar = simpledialog.askstring("Eliminar", "Ingrese el ID del cliente a eliminar:")
        if id_eliminar:
            try:
                if eliminar_registro(id_eliminar):
                    messagebox.showinfo(SUCCESS_TITLE, f"Registro con ID {id_eliminar} eliminado correctamente")
                    self.ver_todos_datos()
                else:
                    messagebox.showwarning("Advertencia", f"No se encontró registro con ID: {id_eliminar}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al eliminar: {str(e)}")
    
    def _obtener_configuracion_campos(self, modo):
        """Obtiene la configuración de campos para el formulario"""
        return {
            'ID': {'tipo': 'texto', 'editable': modo == 'crear'},
            'canceloServicio': {'tipo': 'booleano'},
            'genero': {'tipo': 'opcion', 'opciones': ['Masculino', 'Femenino']},
            'jubiladoMasDeSesenta': {'tipo': 'booleano'},
            'conPareja': {'tipo': 'booleano'},
            'conDependientes': {'tipo': 'booleano'},
            'antiguedadEnMeses': {'tipo': 'numero'},
            'servicioTelefonico': {'tipo': 'opcion', 'opciones': ['No', 'Una línea', 'Múltiples líneas']},
            'servicioDeInternet': {'tipo': 'opcion', 'opciones': ['No', 'DSL', 'Fibra óptica']},
            'seguridadEnLinea': {'tipo': 'booleano'},
            'copiaDeSeguridadEnLinea': {'tipo': 'booleano'},
            'proteccionDeDispositivos': {'tipo': 'booleano'},
            'soporteTecnico': {'tipo': 'booleano'},
            'streamingTV': {'tipo': 'booleano'},
            'streamingDePeliculas': {'tipo': 'booleano'},
            'tipoDeContrato': {'tipo': 'opcion', 'opciones': ['Mes a mes', 'Un año', 'Dos años']},
            'tieneFacturaElectronica': {'tipo': 'booleano'},
            'metodoDePago': {'tipo': 'opcion', 'opciones': ['Cheque enviado por correo', 'Cheque electrónico', 'Tarjeta de crédito (automático)', 'Transferencia bancaria (automático)']},
            'cuentasMensuales': {'tipo': 'numero'},
            'cobroTotal': {'tipo': 'numero'},
        }
    
    def _crear_widget_texto(self, frame_campo, campo, modo, registro_existente, id_registro):
        """Crea un widget de entrada de texto"""
        widget = ttk.Entry(frame_campo)
        if modo == 'actualizar' and campo == 'ID':
            widget.insert(0, id_registro)
            widget['state'] = 'disabled'
        elif registro_existente:
            widget.insert(0, registro_existente.get(campo, ''))
        widget.pack(side='left', fill='x', expand=True)
        return widget
    
    def _crear_widget_numero(self, frame_campo, campo, registro_existente):
        """Crea un widget de entrada numérica"""
        widget = ttk.Entry(frame_campo)
        if registro_existente:
            widget.insert(0, registro_existente.get(campo, ''))
        widget.pack(side='left', fill='x', expand=True)
        return widget
    
    def _crear_widget_booleano(self, frame_campo, campo, registro_existente):
        """Crea un widget de selección booleana (Sí/No)"""
        widget = ttk.Combobox(frame_campo, values=['Sí', 'No'], state='readonly', width=15)
        if registro_existente:
            valor = registro_existente.get(campo, 'No')
            widget.set('Sí' if valor in ['True', 'true', 'Sí', '1'] else 'No')
        else:
            widget.set('No')
        widget.pack(side='left')
        return widget
    
    def _crear_widget_opcion(self, frame_campo, campo, config, registro_existente):
        """Crea un widget de selección de opciones"""
        widget = ttk.Combobox(frame_campo, values=config['opciones'], state='readonly', width=25)
        if registro_existente:
            widget.set(registro_existente.get(campo, config['opciones'][0]))
        else:
            widget.set(config['opciones'][0])
        widget.pack(side='left')
        return widget
    
    def _crear_widget_campo(self, frame_campo, campo, config, modo, registro_existente, id_registro):
        """Crea el widget apropiado para un campo específico"""
        label = ttk.Label(frame_campo, text=self.nombres_columnas.get(campo, campo), width=20)
        label.pack(side='left')
        
        tipo = config['tipo']
        
        if tipo == 'texto':
            return self._crear_widget_texto(frame_campo, campo, modo, registro_existente, id_registro)
        elif tipo == 'numero':
            return self._crear_widget_numero(frame_campo, campo, registro_existente)
        elif tipo == 'booleano':
            return self._crear_widget_booleano(frame_campo, campo, registro_existente)
        elif tipo == 'opcion':
            return self._crear_widget_opcion(frame_campo, campo, config, registro_existente)
        
        return None
    
    def _extraer_datos_widgets(self, widgets, campos):
        """Extrae y convierte los datos de los widgets"""
        datos = {}
        for campo, widget in widgets.items():
            valor = widget.get()
            
            if campos[campo]['tipo'] == 'numero':
                datos[campo] = float(valor)
            elif campos[campo]['tipo'] == 'booleano':
                datos[campo] = valor == 'Sí'
            else:
                datos[campo] = valor
        return datos
    
    def _guardar_registro(self, datos, modo, id_registro, ventana):
        """Guarda el registro (creación o actualización)"""
        try:
            if modo == 'crear':
                id_nuevo = datos['ID']
                if leer_registro(id_nuevo):
                    messagebox.showerror("Error", f"Ya existe un registro con ID: {id_nuevo}")
                    return False
                crear_registro(datos)
                messagebox.showinfo(SUCCESS_TITLE, "Registro creado exitosamente")
            else:
                actualizar_registro(id_registro, datos)
                messagebox.showinfo(SUCCESS_TITLE, "Registro actualizado exitosamente")
            
            ventana.destroy()
            self.ver_todos_datos()
            return True
            
        except ValueError as e:
            messagebox.showerror("Error", f"Error en los datos: {str(e)}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {str(e)}")
            return False
    
    def _ventana_registro(self, modo, registro_existente=None, id_registro=None):
        """Crea ventana para crear o actualizar registros"""
        ventana = tk.Toplevel(self.root)
        ventana.title("Crear Registro" if modo == 'crear' else f"Actualizar Registro {id_registro}")
        ventana.geometry("600x700")
        
        campos = self._obtener_configuracion_campos(modo)
        widgets = {}
        
        for campo, config in campos.items():
            frame_campo = ttk.Frame(ventana)
            frame_campo.pack(fill='x', padx=10, pady=5)
            
            widget = self._crear_widget_campo(frame_campo, campo, config, modo, registro_existente, id_registro)
            if widget:
                widgets[campo] = widget
        
        def guardar():
            datos = self._extraer_datos_widgets(widgets, campos)
            self._guardar_registro(datos, modo, id_registro, ventana)
        
        frame_botones = ttk.Frame(ventana)
        frame_botones.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(frame_botones, text="Guardar", command=guardar).pack(side='left', padx=5)
        ttk.Button(frame_botones, text="Cancelar", command=ventana.destroy).pack(side='left', padx=5)
    
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
        self.model_text.insert(tk.END, "Esto puede tardar unos minutos (selección de features, k-fold y sintonía)...\n")
        
        self._run_thread_safe(self.ejecutar_pipeline)
    
    def ejecutar_pipeline(self):
        """Ejecuta el pipeline completo de ML"""
        try:
            resultados = ejecutar_pipeline_completo()
            
            # Guardar resultados para visualización
            self.resultados_modelos = resultados
            
            self.model_text.delete(1.0, tk.END)
            self.model_text.insert(tk.END, formatear_reporte_pipeline(resultados))
            
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
        ventana.geometry("1100x720")
        
        # Manejar cierre de esta ventana hija
        ventana.protocol("WM_DELETE_WINDOW", lambda: self._cerrar_ventana_segura(ventana))
        
        # Notebook para pestañas de visualización
        notebook = ttk.Notebook(ventana)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self._crear_pestana_sintonia_sensibilidad(notebook)
        self._crear_pestana_comparacion_accuracy(notebook)
        self._crear_pestana_matrices_confusion(notebook)
        self._crear_pestana_metricas_detalladas(notebook)
        self._crear_pestana_validacion_cruzada(notebook)
        self._crear_pestana_features(notebook)
    
    def _cerrar_ventana_segura(self, ventana):
        """Cierra una ventana de forma segura limpiando recursos"""
        try:
            ventana.destroy()
        except tk.TclError:
            pass
    
    def _etiquetar_barras(self, grupos_barras):
        for bars in grupos_barras:
            for bar in bars:
                height = bar.get_height()
                ax_parent = bar.axes
                ax_parent.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8,
                )

    def _crear_pestana_sintonia_sensibilidad(self, notebook):
        """Compara F1 (métrica de sintonía) con sensibilidad, en CV y en test."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Sintonía y Sensibilidad")

        resultados = self.resultados_modelos
        bloques = [
            resultados['resultados_log'],
            resultados['resultados_rf'],
            resultados['resultados_xgb'],
        ]
        modelos = [MODELO_REGRESION_LOGISTICA, MODELO_RANDOM_FOREST, MODELO_XGBOOST]
        x = np.arange(len(modelos))
        width = 0.35

        fig, (ax_cv, ax_test) = plt.subplots(1, 2, figsize=(12, 5.5))

        f1_cv = [b['sintonia']['f1'] for b in bloques]
        sens_cv = [b['sintonia']['sensibilidad'] for b in bloques]
        bars_f1_cv = ax_cv.bar(x - width / 2, f1_cv, width, label='F1 (sintonía)', color='#6a3d9a')
        bars_sens_cv = ax_cv.bar(x + width / 2, sens_cv, width, label=SENSIBILIDAD_RECALL, color='#ff7f0e')
        ax_cv.set_title('Durante la sintonía (CV de la búsqueda)')
        ax_cv.set_ylabel('Score')
        ax_cv.set_xticks(x)
        ax_cv.set_xticklabels(modelos, rotation=15, ha='right')
        ax_cv.set_ylim(0, 1)
        ax_cv.legend()
        self._etiquetar_barras([bars_f1_cv, bars_sens_cv])

        f1_test = [b['metricas']['f1'] for b in bloques]
        sens_test = [b['metricas']['sensibilidad'] for b in bloques]
        bars_f1_test = ax_test.bar(x - width / 2, f1_test, width, label='F1 (sintonía, test)', color='#6a3d9a')
        bars_sens_test = ax_test.bar(x + width / 2, sens_test, width, label='Sensibilidad (Recall, test)', color='#ff7f0e')
        ax_test.set_title('En el conjunto de prueba')
        ax_test.set_ylabel('Score')
        ax_test.set_xticks(x)
        ax_test.set_xticklabels(modelos, rotation=15, ha='right')
        ax_test.set_ylim(0, 1)
        ax_test.legend()
        self._etiquetar_barras([bars_f1_test, bars_sens_test])

        fig.suptitle('F1 (métrica con la que se sintoniza) vs Sensibilidad (Recall de abandonadores)')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _crear_pestana_comparacion_accuracy(self, notebook):
        """Crea pestaña con comparación de accuracy entre modelos"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Comparación Exactitud (Accuracy)")
        
        resultados = self.resultados_modelos
        accuracy_log = resultados['resultados_log']['accuracy']
        accuracy_rf = resultados['resultados_rf']['accuracy']
        accuracy_xgb = resultados['resultados_xgb']['accuracy']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        modelos = [MODELO_REGRESION_LOGISTICA, MODELO_RANDOM_FOREST, MODELO_XGBOOST]
        accuracies = [accuracy_log, accuracy_rf, accuracy_xgb]
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(modelos, accuracies, color=colores)
        ax.set_title('Comparación de Exactitud (Accuracy) en test')
        ax.set_ylabel('Exactitud (Accuracy)')
        ax.set_ylim(0, 1)
        
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
        
        fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
        matrices = [
            (resultados['resultados_log']['confusion_matrix'], MODELO_REGRESION_LOGISTICA, 'Blues'),
            (resultados['resultados_rf']['confusion_matrix'], MODELO_RANDOM_FOREST, 'Oranges'),
            (resultados['resultados_xgb']['confusion_matrix'], MODELO_XGBOOST, 'Greens'),
        ]
        for ax, (cm, titulo, cmap) in zip(ejes, matrices):
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax)
            ax.set_title(titulo)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Realidad')
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels([LABEL_NO_CANCELADO, LABEL_CANCELADO])
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels([LABEL_NO_CANCELADO, LABEL_CANCELADO])
        
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
        m_log = resultados['resultados_log']['metricas']
        m_rf = resultados['resultados_rf']['metricas']
        m_xgb = resultados['resultados_xgb']['metricas']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        nombres = ['Precisión (Precision)', SENSIBILIDAD_RECALL, 'Especificidad (Specificity)', 'F1 (sintonía)']
        x = np.arange(len(nombres))
        width = 0.25
        
        metricas_log = [m_log['precision'], m_log['sensibilidad'], m_log['especificidad'], m_log['f1']]
        metricas_rf = [m_rf['precision'], m_rf['sensibilidad'], m_rf['especificidad'], m_rf['f1']]
        metricas_xgb = [m_xgb['precision'], m_xgb['sensibilidad'], m_xgb['especificidad'], m_xgb['f1']]
        
        bars1 = ax.bar(x - width, metricas_log, width, label=MODELO_REGRESION_LOGISTICA, color='#1f77b4')
        bars2 = ax.bar(x, metricas_rf, width, label=MODELO_RANDOM_FOREST, color='#ff7f0e')
        bars3 = ax.bar(x + width, metricas_xgb, width, label=MODELO_XGBOOST, color='#2ca02c')
        
        ax.set_title('Métricas en test (clase abandonador)')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres)
        ax.legend()
        ax.set_ylim(0, 1)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _crear_pestana_validacion_cruzada(self, notebook):
        """Compara métricas medias de k-fold entre modelos"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Validación Cruzada")
        
        resultados = self.resultados_modelos
        nombres = ['F1 (sintonía)', SENSIBILIDAD_RECALL, 'Precisión (Precision)', 'ROC-AUC']
        claves = ['f1', 'sensibilidad', 'precision', 'roc_auc']
        x = np.arange(len(nombres))
        width = 0.25
        
        def medias(bloque):
            return [bloque['cv'][k]['media'] for k in claves]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, medias(resultados['resultados_log']), width,
                       label=MODELO_REGRESION_LOGISTICA, color='#1f77b4')
        bars2 = ax.bar(x, medias(resultados['resultados_rf']), width,
                       label=MODELO_RANDOM_FOREST, color='#ff7f0e')
        bars3 = ax.bar(x + width, medias(resultados['resultados_xgb']), width,
                       label=MODELO_XGBOOST, color='#2ca02c')
        
        ax.set_title(f'Media {N_FOLDS}-fold sobre entrenamiento (modelos sintonizados)')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres)
        ax.legend()
        ax.set_ylim(0, 1)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def _crear_pestana_features(self, notebook):
        """Muestra el ranking univariado y las variables seleccionadas"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Features")
        
        ranking = self.resultados_modelos['seleccion']['ranking']
        seleccionadas = set(self.resultados_modelos['seleccion']['columnas'])
        ranking = ranking.copy()
        ranking['seleccionada'] = ranking['variable'].isin(seleccionadas)
        
        # Renombrar variables a nombres amigables
        ranking['variable'] = ranking['variable'].apply(lambda x: obtener_nombre_amigable(x, es_procesada=True))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colores = ['#2ca02c' if sel else '#9e9e9e' for sel in ranking['seleccionada']]
        ax.barh(ranking['variable'][::-1], ranking['f_score'][::-1], color=colores[::-1])
        ax.set_title('Ranking F-score (verde = seleccionada por SelectFromModel)')
        ax.set_xlabel('F-score (ANOVA)')
        plt.tight_layout()
        
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

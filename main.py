import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
from crud_service import (
    crear_csv_si_no_existe, obtener_datos, crear_registro, 
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
from visualizaciones import dashboard_completo
import threading

class TelecomAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Telecom-AI - Sistema de Análisis de Clientes")
        self.root.geometry("1200x800")
        
        # Crear el CSV si no existe
        crear_csv_si_no_existe()
        
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear las diferentes pestañas
        self.crear_pestana_gestion_datos()
        self.crear_pestana_estadisticas()
        self.crear_pestana_modelos()
        self.crear_pestana_visualizaciones()
        
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
        
        # Área de texto para mostrar datos
        self.text_area = tk.Text(main_frame, wrap='word', height=20)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        
        self.text_area.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
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
        
        # Área de texto para resultados de modelos
        self.model_text = tk.Text(main_frame, wrap='word', height=20)
        model_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.model_text.yview)
        self.model_text.configure(yscrollcommand=model_scrollbar.set)
        
        self.model_text.pack(side='left', fill='both', expand=True)
        model_scrollbar.pack(side='right', fill='y')
        
    def crear_pestana_visualizaciones(self):
        """Crea la pestaña de visualizaciones"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Visualizaciones")
        
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Botones de visualizaciones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Dashboard Completo", command=self.ejecutar_dashboard_thread).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Matriz de Correlación", command=self.mostrar_matriz_correlacion).pack(side='left', padx=5)
        
        # Área de texto para información
        self.viz_text = tk.Text(main_frame, wrap='word', height=20)
        viz_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.viz_text.yview)
        self.viz_text.configure(yscrollcommand=viz_scrollbar.set)
        
        self.viz_text.pack(side='left', fill='both', expand=True)
        viz_scrollbar.pack(side='right', fill='y')
    
    # Métodos de gestión de datos
    def ver_todos_datos(self):
        """Muestra todos los datos del CSV"""
        try:
            datos = obtener_datos()
            self.text_area.delete(1.0, tk.END)
            
            if datos:
                # Mostrar encabezados
                headers = list(datos[0].keys())
                self.text_area.insert(tk.END, " | ".join(headers) + "\n")
                self.text_area.insert(tk.END, "-" * 100 + "\n")
                
                # Mostrar datos
                for registro in datos:
                    self.text_area.insert(tk.END, " | ".join([str(registro[h]) for h in headers]) + "\n")
                
                self.text_area.insert(tk.END, f"\nTotal de registros: {len(datos)}")
            else:
                self.text_area.insert(tk.END, "No hay datos en el archivo.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer datos: {str(e)}")
    
    def buscar_por_id(self):
        """Busca un registro por ID"""
        id_buscar = simpledialog.askstring("Buscar", "Ingrese el ID del cliente:")
        if id_buscar:
            try:
                registro = leer_registro(id_buscar)
                self.text_area.delete(1.0, tk.END)
                
                if registro:
                    for key, value in registro.items():
                        self.text_area.insert(tk.END, f"{key}: {value}\n")
                else:
                    self.text_area.insert(tk.END, f"No se encontró registro con ID: {id_buscar}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al buscar: {str(e)}")
    
    def crear_registro_gui(self):
        """Interfaz para crear un nuevo registro"""
        # Implementación simplificada - podría mejorarse con un formulario más completo
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Función de crear registro - Implementación pendiente\n")
        self.text_area.insert(tk.END, "Para crear un registro manualmente, editar el CSV directamente\n")
    
    def actualizar_registro_gui(self):
        """Interfaz para actualizar un registro"""
        id_actualizar = simpledialog.askstring("Actualizar", "Ingrese el ID del cliente a actualizar:")
        if id_actualizar:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, f"Función de actualizar registro para ID: {id_actualizar}\n")
            self.text_area.insert(tk.END, "Implementación pendiente - editar CSV directamente\n")
    
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
            counts = distribucion_cancelacion()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Distribución de Cancelación:\n")
            self.stats_text.insert(tk.END, str(counts))
        except Exception as e:
            messagebox.showerror("Error", f"Error en distribución: {str(e)}")
    
    def mostrar_grafico_genero(self):
        """Muestra gráfico por género"""
        try:
            fig = grafico_cancelacion_por_genero()
            fig.show()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por género generado en ventana emergente")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    def mostrar_grafico_contrato(self):
        """Muestra gráfico por contrato"""
        try:
            fig = grafico_cancelacion_por_contrato()
            fig.show()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por contrato generado en ventana emergente")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    def mostrar_grafico_pago(self):
        """Muestra gráfico por método de pago"""
        try:
            fig = grafico_cancelacion_por_pago()
            fig.show()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por método de pago generado en ventana emergente")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    def mostrar_grafico_antiguedad(self):
        """Muestra gráfico por antigüedad"""
        try:
            fig = grafico_cancelacion_por_antiguedad()
            fig.show()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Gráfico por antigüedad generado en ventana emergente")
        except Exception as e:
            messagebox.showerror("Error", f"Error en gráfico: {str(e)}")
    
    # Métodos de modelos
    def ejecutar_pipeline_thread(self):
        """Ejecuta el pipeline de ML en un hilo separado"""
        self.model_text.delete(1.0, tk.END)
        self.model_text.insert(tk.END, "Ejecutando pipeline de Machine Learning...\n")
        self.model_text.insert(tk.END, "Esto puede tomar varios segundos...\n")
        
        thread = threading.Thread(target=self.ejecutar_pipeline)
        thread.start()
    
    def ejecutar_pipeline(self):
        """Ejecuta el pipeline completo de ML"""
        try:
            resultados = ejecutar_pipeline_completo()
            
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
    
    # Métodos de visualizaciones
    def ejecutar_dashboard_thread(self):
        """Ejecuta el dashboard en un hilo separado"""
        self.viz_text.delete(1.0, tk.END)
        self.viz_text.insert(tk.END, "Generando dashboard completo...\n")
        self.viz_text.insert(tk.END, "Esto puede tomar varios segundos...\n")
        
        thread = threading.Thread(target=self.ejecutar_dashboard)
        thread.start()
    
    def ejecutar_dashboard(self):
        """Ejecuta el dashboard completo"""
        try:
            resultados = dashboard_completo()
            
            self.viz_text.delete(1.0, tk.END)
            self.viz_text.insert(tk.END, "=== Dashboard Completado ===\n\n")
            self.viz_text.insert(tk.END, "Se han generado las siguientes visualizaciones:\n")
            self.viz_text.insert(tk.END, "- Matriz de correlación\n")
            self.viz_text.insert(tk.END, "- Distribución de cancelación\n")
            self.viz_text.insert(tk.END, "- Boxplots de variables numéricas\n")
            self.viz_text.insert(tk.END, "- Scatter plot de antigüedad vs cuentas mensuales\n")
            self.viz_text.insert(tk.END, "- Gráficos categóricos\n")
            self.viz_text.insert(tk.END, "- Gráficos de costos\n\n")
            self.viz_text.insert(tk.END, "Las visualizaciones se mostraron en ventanas emergentes.")
            
            messagebox.showinfo("Éxito", "Dashboard completado exitosamente")
        except Exception as e:
            self.viz_text.insert(tk.END, f"\nError: {str(e)}")
            messagebox.showerror("Error", f"Error en dashboard: {str(e)}")
    
    def mostrar_matriz_correlacion(self):
        """Muestra la matriz de correlación"""
        try:
            from modelo_service import preparar_datos_ml
            from visualizaciones import mostrar_matriz_correlacion
            
            X, y = preparar_datos_ml()
            mostrar_matriz_correlacion(X, y)
            
            self.viz_text.delete(1.0, tk.END)
            self.viz_text.insert(tk.END, "Matriz de correlación generada en ventana emergente")
        except Exception as e:
            messagebox.showerror("Error", f"Error en matriz de correlación: {str(e)}")

def main():
    root = tk.Tk()
    app = TelecomAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

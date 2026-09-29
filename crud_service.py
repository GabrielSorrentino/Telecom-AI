import csv
import os
import shutil

FILE_BASE = 'Telecom_base.csv'
FILE_TRABAJO = 'Telecom_AI.csv'
FIELD_NAMES = ['ID', 'canceloServicio', 'genero', 'jubiladoMasDeSesenta', 'conPareja', 'conDependientes', 'antiguedadEnMeses', 'servicioTelefonico', 'servicioDeInternet', 'seguridadEnLinea', 'copiaDeSeguridadEnLinea', 'proteccionDeDispositivos', 'soporteTecnico', 'streamingTV', 'streamingDePeliculas', 'tipoDeContrato', 'tieneFacturaElectronica', 'metodoDePago', 'cuentasMensuales', 'cobroTotal']

def inicializar_csv_trabajo():
    """Inicializa el CSV de trabajo desde el archivo base"""
    if not os.path.exists(FILE_BASE):
        raise FileNotFoundError(f"El archivo base {FILE_BASE} no existe")
    
    if not os.path.exists(FILE_TRABAJO):
        shutil.copy2(FILE_BASE, FILE_TRABAJO)
        print(f"CSV de trabajo creado desde {FILE_BASE}")
    else:
        print(f"Usando CSV de trabajo existente: {FILE_TRABAJO}")

def obtener_datos():
    with open(FILE_TRABAJO, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

def crear_registro(data):
    with open(FILE_TRABAJO, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writerow(data)

def leer_registro(id_registro):
    datos = obtener_datos()
    for registro in datos:
        if registro['ID'] == id_registro:
            return registro
    return None

def actualizar_registro(id_registro, nuevos_datos):
    datos = obtener_datos()
    registros_actualizados = []
    encontrado = False
    
    for registro in datos:
        if registro['ID'] == id_registro:
            registro.update(nuevos_datos)
            encontrado = True
        registros_actualizados.append(registro)
    
    if encontrado:
        with open(FILE_TRABAJO, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(registros_actualizados)
        return True
    return False

def eliminar_registro(id_registro):
    datos = obtener_datos()
    registros_filtrados = [reg for reg in datos if reg['ID'] != id_registro]
    
    if len(registros_filtrados) < len(datos):
        with open(FILE_TRABAJO, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(registros_filtrados)
        return True
    return False

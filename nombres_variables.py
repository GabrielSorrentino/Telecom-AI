"""Mapeo de nombres de variables técnicas a nombres amigables para el usuario"""

# Mapeo de nombres de columnas del CSV original a nombres amigables
NOMBRES_AMIGABLES_ORIGINALES = {
    'antiguedadEnMeses': 'Antigüedad (meses)',
    'cuentasMensuales': 'Cuentas mensuales ($)',
    'cobroTotal': 'Cobro total ($)',
    'canceloServicio': 'Canceló el servicio',
    'genero': 'Género',
    'jubiladoMasDeSesenta': 'Jubilado (+60)',
    'conPareja': 'Con pareja',
    'conDependientes': 'Con dependientes',
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
}

# Mapeo de nombres de variables procesadas (después del encoding) a nombres amigables
NOMBRES_AMIGABLES_PROCESADAS = {
    'MultiplesLineas': 'Múltiples líneas telefónicas',
    'UnicaLinea': 'Única línea telefónica',
    'TieneDSL': 'Tiene DSL',
    'TieneFibraOptica': 'Tiene Fibra Óptica',
    'ContratoDosAnos': 'Contrato de 2 años',
    'ContratoUnAno': 'Contrato de 1 año',
    'PagoChequeElectronico': 'Pago: Cheque electrónico',
    'PagoTarjetaCredito': 'Pago: Tarjeta de crédito',
    'PagoTransferenciaBancaria': 'Pago: Transferencia bancaria',
    'antiguedadEnMeses': 'Antigüedad (meses)',
    'cuentasMensuales': 'Cuentas mensuales ($)',
    'jubiladoMasDeSesenta': 'Jubilado (+60)',
    'conPareja': 'Con pareja',
    'conDependientes': 'Con dependientes',
    'seguridadEnLinea': 'Seguridad en línea',
    'copiaDeSeguridadEnLinea': 'Copia de seguridad',
    'proteccionDeDispositivos': 'Protección de dispositivos',
    'soporteTecnico': 'Soporte técnico',
    'streamingTV': 'Streaming TV',
    'streamingDePeliculas': 'Streaming películas',
    'tieneFacturaElectronica': 'Factura electrónica',
    'canceloServicio': 'Canceló el servicio',
}

def obtener_nombre_amigable(variable_tecnica, es_procesada=False):
    """
    Obtiene el nombre amigable de una variable técnica.
    
    Args:
        variable_tecnica: Nombre técnico de la variable
        es_procesada: True si es una variable procesada (después del encoding)
    
    Returns:
        Nombre amigable o el nombre original si no hay mapeo
    """
    mapeo = NOMBRES_AMIGABLES_PROCESADAS if es_procesada else NOMBRES_AMIGABLES_ORIGINALES
    return mapeo.get(variable_tecnica, variable_tecnica)

def renombrar_columnas_dataframe(df, es_procesado=False):
    """
    Renombra las columnas de un DataFrame usando el mapeo de nombres amigables.
    
    Args:
        df: DataFrame con nombres técnicos
        es_procesado: True si son columnas procesadas (después del encoding)
    
    Returns:
        DataFrame con columnas renombradas
    """
    mapeo = NOMBRES_AMIGABLES_PROCESADAS if es_procesado else NOMBRES_AMIGABLES_ORIGINALES
    return df.rename(columns=mapeo)

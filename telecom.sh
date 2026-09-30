#!/bin/bash
# Script para Linux/macOS - Ejecutar Telecom-AI con Docker

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker no está instalado o no está en ejecución."
    echo "Por favor instale Docker desde https://docs.docker.com/get-docker/"
    exit 1
fi

# Obtener IDs del usuario actual
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Configurar X11 según sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    X11_PARAMS="--env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    X11_PARAMS="--env DISPLAY=host.docker.internal:0"
else
    echo "Sistema operativo no soportado: $OSTYPE"
    exit 1
fi

# Construir imagen
echo "Construyendo imagen desde Dockerfile..."
docker build -t telecom-ai . > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Falló la construcción de la imagen Docker"
    exit 1
fi

# Ejecutar contenedor
echo "Iniciando contenedor Telecom-AI..."
docker run --rm \
    --user "$USER_ID:$GROUP_ID" \
    $X11_PARAMS \
    --volume "$(pwd):/app" \
    telecom-ai

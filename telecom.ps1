# Script para Windows 10/11 - Ejecutar Telecom-AI con Docker

# Verificar si Docker está instalado
try {
    docker version > $null 2>&1
} catch {
    Write-Host "ERROR: Docker no está instalado o no está en ejecución."
    Write-Host "Por favor instale Docker Desktop desde https://www.docker.com/products/docker-desktop"
    exit 1
}

# Construir imagen
Write-Host "Construyendo imagen desde Dockerfile..."
docker build -t telecom-ai . > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Falló la construcción de la imagen Docker"
    exit 1
}

# Ejecutar contenedor
Write-Host "Iniciando contenedor Telecom-AI..."
docker run --rm `
    --volume "${PWD}:/app" `
    --env DISPLAY=host.docker.internal:0 `
    telecom-ai

if ($LASTEXITCODE -ne 0) {
    Write-Host "El contenedor terminó con errores. Código: $LASTEXITCODE"
}

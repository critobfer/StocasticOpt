#!/bin/bash

# Actualizar los paquetes del sistema
sudo apt-get update

# Instalar dependencias
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Crear un entorno virtual con Python 3.12
python3.12 -m venv venv

# Activar el entorno virtual
source venv/bin/activate

# Instalar dependencias del proyecto
pip install -r requirements.txt

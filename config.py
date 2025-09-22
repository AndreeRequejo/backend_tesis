# config.py - Configuración de la API

# Nombres de las clases
CLASS_NAMES = [
    "Mild Acne",      # Clase 0 - Acné leve
    "Moderate Acne",  # Clase 1 - Acné moderado
    "Severe Acne",    # Clase 2 - Acné severo
    "Very Severe"     # Clase 3 - Muy severo
]

# Configuración del modelo
MODEL_PATH = "acne_model.pt"

# Configuración de imagen
IMAGE_SIZE = (224, 224)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# Configuración de la API
API_TITLE = "Documentación Tesis Acne"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# API de Clasificación de Severidad de Acné

Esta API utiliza deep learning para clasificar la severidad del acné en imágenes faciales.

## Funcionalidades
* **Predicción Individual**: `/predict`
* **Predicción en Lote**: `/predict/batch`

## Clases
- **0: Acné leve
- **1: Acné moderado
- **2: Acné severo
- **3: Acné muy severo

## Formatos Soportados
JPG, PNG, JPEG
"""

# Configuración del servidor
HOST = "127.0.0.1"
PORT = 8000

# Límites
MAX_BATCH_SIZE = 3
VALID_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
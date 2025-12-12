# config.py - Configuración de la API

# Nombres de las clases
CLASS_NAMES = [
    "Leve",      # Clase 0 - Acné leve
    "Moderado",  # Clase 1 - Acné moderado
    "Severo",    # Clase 2 - Acné severo
    #"Muy Severo" # Clase 3 - Muy severo
]

# Configuración del modelo
MODEL_PATH = "core/acne_model.pt"
#MODEL_PATH = "core/efficient.pt"

# Configuración del modelo ONNX para detección de anime/3D
ONNX_MODEL_PATH = "core/model.onnx"
ANIME_CLASS_NAMES = ["anime", "real"]  # Clases del modelo ONNX
REAL_CLASS_CONFIDENCE_THRESHOLD = 0.93  # 93% de confianza mínima para imágenes reales
ONNX_IMAGE_SIZE = 384  # Tamaño de entrada para el modelo ONNX (384x384)

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

## Flujo de Validación:
Las imágenes pasan por un flujo de filtros secuencial antes de la clasificación final:

**1. MediaPipe (Filtro Inicial)**
- Detecta presencia de rostros (≥80% confianza)  
- Si NO hay rostros → RECHAZA inmediatamente
- Si hay MÚLTIPLES rostros → RECHAZA inmediatamente
- Si hay EXACTAMENTE 1 rostro → Continúa al siguiente filtro

**2. ONNX Model (Filtro de Autenticidad)**
- Solo procesa si hay exactamente 1 rostro detectado
- Valida que sea imagen REAL vs ANIMADA/3D (≥97% confianza para imágenes reales)
- Si detecta ANIMADO → RECHAZA
- Si detecta REAL → APRUEBA para clasificación de acné

## Clases
- **0: Acné leve
- **1: Acné moderado
- **2: Acné severo

## Formatos Soportados
JPG, PNG, JPEG, WEBP

## Requisitos de Imagen
- Debe contener exactamente un rostro visible
- La imagen debe ser una fotografía real (no animada o 3D)
"""

# Configuración del servidor
HOST = "127.0.0.1"
PORT = 8080

# Límites
MAX_BATCH_SIZE = 3
VALID_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

# Configuración del detector de rostros
FACE_DETECTOR_CONFIG = {
    "use_mtcnn": True,  # True para MTCNN, False para MediaPipe
    "mediapipe_confidence": 0.8,  # Confianza mínima para MediaPipe
    "mtcnn_confidence": 0.8,  # Confianza mínima para MTCNN
}

# Configuración MTCNN para validación de rostros
MTCNN_CONFIG = {
    "image_size": 160,  # Tamaño estándar para MTCNN  
    "margin": 0,  # Margen alrededor del rostro
    "min_face_size": 20,  # Tamaño mínimo de rostro en píxeles
    "thresholds": [0.6, 0.7, 0.7],  # Umbrales para las 3 redes de MTCNN
    "factor": 0.709,  # Factor de escalado para pirámide de imágenes
    "post_process": True,  # Aplicar post-procesamiento
    "select_largest": True,  # Seleccionar el rostro más grande si hay múltiples
    "keep_all": False,  # Solo necesitamos saber si hay rostros, no extraerlos todos
}

# Mensajes de validación de rostros
FACE_VALIDATION_MESSAGES = {
    "no_face": "No se detectó ningún rostro en la imagen. Por favor, ingrese una imagen que contenga al menos un rostro visible.",
    "multiple_faces": "Se detectaron múltiples rostros en la imagen. Por favor, ingrese una imagen que contenga solo un rostro visible.",
    "low_confidence": "No se pudo detectar un rostro con suficiente confianza. Por favor, ingrese otra imagen.",
    "anime_detected": "Se detectó una imagen de personaje animado o 3D. Por favor, utilice una fotografía real de una persona.",
    "low_real_confidence": "La imagen no parece ser una fotografía real. Por favor, utilice una fotografía clara y real de una persona.",
}
# config.py - Configuración de la API

# Nombres de las clases
CLASS_NAMES = [
    "Leve",      # Clase 0 - Acné leve
    "Moderado",  # Clase 1 - Acné moderado
    "Severo",    # Clase 2 - Acné severo
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

# Configuración del servidor
HOST = "127.0.0.1"
PORT = 8080

# Límites
MAX_BATCH_SIZE = 3
VALID_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

# Configuración del detector de rostros
FACE_DETECTOR_CONFIG = {
    "mediapipe_confidence": 0.8,  # Confianza mínima para MediaPipe (filtro inicial)
    "mtcnn_confidence": 0.8,  # Confianza mínima para MTCNN
}

# Configuración MTCNN para validación de rostros
MTCNN_CONFIG = {
    "image_size": 160,  # Tamaño estándar para MTCNN  
    "margin": 0,  # Margen alrededor del rostro
}

# Mensajes de validación de rostros
FACE_VALIDATION_MESSAGES = {
    "no_face": "No se detectó ningún rostro en la imagen. Por favor, ingrese una imagen que contenga al menos un rostro visible.",
    "multiple_faces": "Se detectaron múltiples rostros en la imagen. Por favor, ingrese una imagen que contenga solo un rostro visible.",
    "low_confidence": "No se pudo detectar un rostro con suficiente confianza. Por favor, ingrese otra imagen.",
    "anime_detected": "Se detectó una imagen de personaje animado o 3D. Por favor, utilice una fotografía real de una persona.",
    "low_real_confidence": "La imagen no parece ser una fotografía real. Por favor, utilice una fotografía clara y real de una persona.",
}

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
- Valida que sea imagen REAL vs ANIMADA/3D
- Si detecta ANIMADO → RECHAZA
- Si detecta REAL → APRUEBA para clasificación de acné

## Clases
- **0: Acné leve
- **1: Acné moderado
- **2: Acné severo

## Formatos Soportados
JPG, PNG, JPEG, WEBP
"""
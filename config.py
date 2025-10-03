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

# Configuración del modelo ONNX para detección de anime/3D
ONNX_MODEL_PATH = "core/model.onnx"
ANIME_CLASS_NAMES = ["anime", "real"]  # Clases del modelo ONNX
REAL_CLASS_CONFIDENCE_THRESHOLD = 0.97  # 97% de confianza mínima para imágenes reales
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

## Validación de Rostros y Calidad
Todas las imágenes son validadas usando MediaPipe y métricas de calidad balanceadas para garantizar que:
- Contengan al menos un rostro visible con buena confianza (≥80%)
- El rostro sea de tamaño suficiente (≥40x40 píxeles)
- La imagen tenga calidad aceptable (solo rechaza imágenes extremadamente borrosas)
- El rostro ocupe una proporción razonable de la imagen
- La resolución de imagen sea adecuada (≥80x80 píxeles)

## Validación Inteligente
- **Modo Balanceado**: Validaciones menos invasivas por defecto
- **Detección de Bordes**: Validación secundaria para imágenes con desenfoque moderado
- **Parámetro Opcional**: `skip_quality_check` para omitir validaciones de calidad

## Clases
- **0: Acné leve
- **1: Acné moderado
- **2: Acné severo

## Formatos Soportados
JPG, PNG, JPEG

## Requisitos de Imagen (Relajados)
- Debe contener al menos un rostro visible
- Rostro de tamaño mínimo 40x40 píxeles
- Resolución mínima 80x80 píxeles
- El rostro debe ocupar al menos 1.5% del área total de la imagen
- Solo se rechazan imágenes extremadamente borrosas o de muy baja calidad
"""

# Configuración del servidor
HOST = "127.0.0.1"
PORT = 8080

# Límites
MAX_BATCH_SIZE = 3
VALID_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

# Configuración MTCNN para validación de rostros
MTCNN_CONFIG = {
    "min_face_size": 20,  # Tamaño mínimo de rostro en píxeles
    "thresholds": [0.6, 0.7, 0.7],  # Umbrales para las 3 redes de MTCNN
    "factor": 0.709,  # Factor de escalado para pirámide de imágenes
    "post_process": True,  # Aplicar post-procesamiento
    "select_largest": True,  # Seleccionar el rostro más grande si hay múltiples
    "keep_all": False,  # Solo necesitamos saber si hay rostros, no extraerlos todos
}

# Configuración para validación de calidad de imagen
IMAGE_QUALITY_CONFIG = {
    "min_confidence": 0.8,  # Confianza mínima del detector MTCNN (reducido de 0.85)
    "min_face_size": 40,  # Tamaño mínimo del rostro en píxeles (reducido de 50)
    "min_image_size": 80,  # Tamaño mínimo de la imagen en píxeles (reducido de 100)
    "max_blur_threshold": 50,  # Umbral máximo de desenfoque - más permisivo (reducido de 100)
    "min_brightness": 20,  # Brillo mínimo (0-255) - más permisivo (reducido de 30)
    "max_brightness": 240,  # Brillo máximo (0-255) - más permisivo (aumentado de 230)
    "min_contrast": 15,  # Contraste mínimo - más permisivo (reducido de 20)
    "min_face_area_ratio": 0.015,  # Ratio mínimo del área del rostro - más permisivo (reducido de 0.02)
    "enable_quality_checks": True,  # Permitir desactivar validaciones de calidad
    "strict_mode": False,  # Modo estricto desactivado por defecto
}

# Mensajes de validación de rostros y calidad
FACE_VALIDATION_MESSAGES = {
    "no_face": "No se detectó ningún rostro en la imagen. Por favor, ingrese una imagen que contenga al menos un rostro visible.",
    "face_too_small": "El rostro detectado es demasiado pequeño. Por favor, ingrese una imagen con un rostro más grande y visible.",
    "multiple_faces": "Se detectaron múltiples rostros. Por favor, utilice una imagen con un solo rostro.",
    "low_confidence": "La calidad de detección del rostro es baja. Por favor, ingrese una imagen más clara y bien iluminada.",
    "blurry_image": "La imagen está demasiado borrosa. Por favor, ingrese una imagen más nítida y enfocada.",
    "poor_lighting": "La iluminación de la imagen es inadecuada (muy oscura o muy brillante). Por favor, ingrese una imagen con mejor iluminación.",
    "low_contrast": "La imagen tiene contraste muy bajo. Por favor, ingrese una imagen con mejor contraste.",
    "image_too_small": "La imagen es demasiado pequeña. Por favor, ingrese una imagen de mayor resolución.",
    "face_too_small_ratio": "El rostro ocupa muy poco espacio en la imagen. Por favor, ingrese una imagen donde el rostro sea más prominente.",
    "anime_detected": "Se detectó una imagen de personaje animado. Por favor, utilice una imagen real de una persona.",
    "low_real_confidence": "La confianza de que esta sea una imagen real es muy baja. Por favor, utilice una imagen clara y real de una persona.",
}
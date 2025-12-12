"""
Utilidades para procesamiento de imágenes
"""
import numpy as np
import io
import logging
from PIL import Image
from torchvision import transforms
from fastapi import UploadFile
from config import *

logger = logging.getLogger(__name__)

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def process_image(image_bytes: bytes):
    """Procesar imagen para predicción"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = transform(image).unsqueeze(0)
    return tensor


def validate_image(file: UploadFile) -> bool:
    """Validar tipo de imagen"""
    return file.content_type in VALID_IMAGE_TYPES if file.content_type else False


def preprocess_image_for_onnx(image_bytes: bytes, onnx_session) -> np.ndarray:
    """
    Preprocesar imagen para el modelo ONNX
    
    Args:
        image_bytes: Bytes de la imagen
        onnx_session: Sesión ONNX
        
    Returns:
        numpy array: Array de imagen preprocesado para ONNX
    """
    # Abrir y convertir imagen
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Obtener las dimensiones esperadas del modelo dinámicamente
    if onnx_session is not None:
        input_shape = onnx_session.get_inputs()[0].shape
        
        # Extraer dimensiones (asumiendo formato NCHW: [batch, channels, height, width])
        if len(input_shape) == 4:
            expected_height = input_shape[2] if isinstance(input_shape[2], int) else ONNX_IMAGE_SIZE
            expected_width = input_shape[3] if isinstance(input_shape[3], int) else ONNX_IMAGE_SIZE
        else:
            expected_height = expected_width = ONNX_IMAGE_SIZE
    else:
        expected_height = expected_width = ONNX_IMAGE_SIZE
    
    # Redimensionar a tamaño esperado por el modelo ONNX
    image = image.resize((expected_width, expected_height))
    
    # Convertir a array numpy y normalizar
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalizar a [0,1]
    
    # Reorganizar dimensiones para ONNX: (batch_size, channels, height, width)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array
"""
Utilidades para procesamiento de imágenes
"""
import numpy as np
import cv2
import io
import logging
from PIL import Image, ImageStat
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


def calculate_image_blur(image_array):
    """
    Calcular el nivel de desenfoque usando múltiples métricas para mayor precisión
    
    Args:
        image_array: Array numpy de la imagen en escala de grises
        
    Returns:
        float: Valor de desenfoque (mayor valor = menos borroso)
    """
    # Método 1: Varianza del Laplaciano (método principal)
    laplacian_var = cv2.Laplacian(image_array, cv2.CV_64F).var()
    
    # Método 2: Gradiente de Sobel para validación adicional
    sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mean = np.mean(sobel_magnitude)
    
    # Combinar ambas métricas para mejor precisión
    # Si la imagen tiene bordes definidos (Sobel alto), es menos probable que esté borrosa
    combined_score = laplacian_var + (sobel_mean * 0.1)
    
    return combined_score


def calculate_image_brightness(image):
    """
    Calcular el brillo promedio de la imagen
    
    Args:
        image: Imagen PIL
        
    Returns:
        float: Brillo promedio (0-255)
    """
    stat = ImageStat.Stat(image)
    return stat.mean[0] if len(stat.mean) == 1 else sum(stat.mean) / len(stat.mean)


def calculate_image_contrast(image):
    """
    Calcular el contraste de la imagen usando la desviación estándar
    
    Args:
        image: Imagen PIL
        
    Returns:
        float: Valor de contraste
    """
    stat = ImageStat.Stat(image)
    return stat.stddev[0] if len(stat.stddev) == 1 else sum(stat.stddev) / len(stat.stddev)


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
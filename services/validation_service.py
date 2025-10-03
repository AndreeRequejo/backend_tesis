"""
Servicio para validaciones de imagen y rostros
"""
import numpy as np
import cv2
import io
import logging
import mediapipe as mp
from PIL import Image
from utils.image_utils import (
    calculate_image_blur, 
    calculate_image_brightness, 
    calculate_image_contrast,
    preprocess_image_for_onnx
)
from config import *

logger = logging.getLogger(__name__)


def validate_real_image(image_bytes: bytes, onnx_session) -> tuple:
    """
    Validar que la imagen sea una foto real de una persona usando modelo ONNX
    
    Args:
        image_bytes: Bytes de la imagen
        onnx_session: Sesión ONNX para validación
        
    Returns:
        tuple: (es_real, confianza, tipo_detectado, mensaje_error)
    """
    if onnx_session is None:
        return False, 0.0, "unknown", "Modelo ONNX no disponible"
    
    try:
        # Preprocesar imagen
        input_array = preprocess_image_for_onnx(image_bytes, onnx_session)
        
        # Obtener información de entrada del modelo ONNX
        input_name = onnx_session.get_inputs()[0].name
        expected_shape = onnx_session.get_inputs()[0].shape
        
        # Verificar compatibilidad de formas antes de la predicción
        if input_array.shape[2] != expected_shape[2] or input_array.shape[3] != expected_shape[3]:
            logger.warning(f"Inconsistencia de dimensiones: esperado {expected_shape}, actual {input_array.shape}")
        
        # Realizar predicción
        outputs = onnx_session.run(None, {input_name: input_array})
        probabilities = outputs[0][0]  # Asumir que la salida son probabilidades
        
        # Aplicar softmax si es necesario (algunos modelos ya lo incluyen)
        if np.sum(probabilities) > 1.01:  # Si la suma no es ~1, aplicar softmax
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        
        # Obtener predicción
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        # Verificar si hay suficientes clases en ANIME_CLASS_NAMES
        if predicted_class >= len(ANIME_CLASS_NAMES):
            logger.warning(f"Clase predicha {predicted_class} fuera de rango. Usando clase 0.")
            predicted_class = 0
            confidence = float(probabilities[0]) if len(probabilities) > 0 else 0.0
        
        predicted_type = ANIME_CLASS_NAMES[predicted_class]
        
        logger.info(f"Detección ONNX - Tipo: {predicted_type}, Confianza: {confidence:.4f}")
        
        # Si la predicción es "real", aplicar umbral estricto del 97%
        if predicted_type == "real":
            if confidence >= REAL_CLASS_CONFIDENCE_THRESHOLD:
                return True, confidence, predicted_type, ""
            else:
                return False, confidence, predicted_type, FACE_VALIDATION_MESSAGES["low_real_confidence"]
        else:
            # Si detecta anime o 3D, rechazar independientemente de la confianza
            return False, confidence, predicted_type, FACE_VALIDATION_MESSAGES["anime_detected"]
    
    except Exception as e:
        logger.error(f"Error en validación de imagen real: {e}")
        return False, 0.0, "error", f"Error al procesar la imagen: {str(e)}"


def validate_image_quality(image_bytes: bytes) -> tuple:
    """
    Validar la calidad general de la imagen con validaciones menos invasivas
    
    Args:
        image_bytes: Bytes de la imagen
        
    Returns:
        tuple: (es_válida, mensaje_error)
    """
    try:
        # Verificar si las validaciones de calidad están habilitadas
        if not IMAGE_QUALITY_CONFIG.get("enable_quality_checks", True):
            return True, ""
        
        # Abrir imagen
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Verificar tamaño mínimo de imagen (validación básica)
        width, height = image.size
        if width < IMAGE_QUALITY_CONFIG["min_image_size"] or height < IMAGE_QUALITY_CONFIG["min_image_size"]:
            return False, FACE_VALIDATION_MESSAGES["image_too_small"]
        
        # Convertir a array numpy para análisis
        image_array = np.array(image)
        
        # Convertir a escala de grises para análisis de desenfoque
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Validar desenfoque (solo rechazar si está MUY borrosa)
        blur_value = calculate_image_blur(gray_image)
        logger.info(f"Valor de desenfoque calculado: {blur_value:.2f} (umbral: {IMAGE_QUALITY_CONFIG['max_blur_threshold']})")
        
        # Solo rechazar si está extremadamente borrosa
        if blur_value < IMAGE_QUALITY_CONFIG["max_blur_threshold"]:
            # Segunda validación: verificar si al menos hay algunos bordes detectables
            edges = cv2.Canny(gray_image, 40, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Si hay suficientes bordes detectados, la imagen probablemente es aceptable
            if edge_density > 0.02:  # Al menos 2% de píxeles son bordes
                logger.info(f"Imagen salvada por detección de bordes: {edge_density:.3f}")
                # Continuar con otras validaciones pero no rechazar por desenfoque
            else:
                return False, FACE_VALIDATION_MESSAGES["blurry_image"]
        
        # En modo no estricto, solo hacer validaciones básicas
        if not IMAGE_QUALITY_CONFIG.get("strict_mode", False):
            return True, ""
        
        # Validaciones adicionales solo en modo estricto
        # Validar brillo extremo
        brightness = calculate_image_brightness(image)
        if brightness < IMAGE_QUALITY_CONFIG["min_brightness"] or brightness > IMAGE_QUALITY_CONFIG["max_brightness"]:
            return False, FACE_VALIDATION_MESSAGES["poor_lighting"]
        
        # Validar contraste muy bajo
        contrast = calculate_image_contrast(image)
        if contrast < IMAGE_QUALITY_CONFIG["min_contrast"]:
            return False, FACE_VALIDATION_MESSAGES["low_contrast"]
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error en validación de calidad de imagen: {e}")
        return False, f"Error al evaluar la calidad de la imagen: {str(e)}"


def validate_face_in_image(image_bytes: bytes, onnx_session) -> tuple:
    """
    Validar que la imagen contenga exactamente un rostro y sea una imagen real
    
    FLUJO:
    1. MediaPipe (FILTRO INICIAL) - Detectar presencia y cantidad de rostros
       → Si NO hay rostros o hay MÚLTIPLES rostros → RECHAZAR inmediatamente
    2. ONNX Model (FILTRO DE AUTENTICIDAD) - Solo si hay exactamente 1 rostro
       → Si es ANIMADO/3D → RECHAZAR
    3. Validaciones de Calidad (FILTRO DE CALIDAD) - Solo si es imagen real
       → Si calidad muy baja → RECHAZAR  
    4. Continuar al modelo principal de clasificación de acné
    
    Args:
        image_bytes: Bytes de la imagen
        onnx_session: Sesión ONNX para validación
    Returns:
        tuple: (es_válida, mensaje_error, confianza_rostro, info_validacion)
    """
    try:
        # PASO 1: MEDIAPIPE - FILTRO INICIAL (MÁS RÁPIDO Y LIGERO)
        # Convertir los bytes a imagen RGB
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"No se pudo abrir la imagen: {e}")
            return False, "No se pudo abrir la imagen. Asegúrate de que el archivo es una imagen válida.", None, {
                "tipo_detectado": "error",
                "confianza_tipo": 0.0,
                "es_real": False
            }

        image_np = np.array(image)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            logger.error("La imagen no tiene 3 canales RGB")
            return False, "La imagen debe tener 3 canales (RGB).", None, {
                "tipo_detectado": "error",
                "confianza_tipo": 0.0,
                "es_real": False
            }

        # Usar mediapipe para detección de rostros (PRIMERA VALIDACIÓN)
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8) as face_detection:
            results = face_detection.process(image_np)
            
            # FILTRO: No hay rostros detectados
            if not results.detections or len(results.detections) == 0:
                logger.info("MediaPipe: No se detectaron rostros - Rechazando sin procesar ONNX")
                return False, FACE_VALIDATION_MESSAGES["no_face"], None, {
                    "tipo_detectado": "no_face",
                    "confianza_tipo": 0.0,
                    "es_real": False
                }
            
            # FILTRO: Múltiples rostros detectados
            if len(results.detections) > 1:
                logger.info(f"MediaPipe: Se detectaron {len(results.detections)} rostros - Rechazando sin procesar ONNX")
                return False, FACE_VALIDATION_MESSAGES["multiple_faces"], None, {
                    "tipo_detectado": "multiple_faces",
                    "confianza_tipo": 0.0,
                    "es_real": False
                }
            
            # Obtener confianza del único rostro detectado
            confianza_rostro = results.detections[0].score[0] if results.detections[0].score else None
            logger.info(f"MediaPipe: 1 rostro detectado con confianza {confianza_rostro:.4f} - Continuando al ONNX")

        # PASO 2: ONNX MODEL - FILTRO DE AUTENTICIDAD (Solo si hay exactamente 1 rostro)
        is_real, real_confidence, detected_type, real_error = validate_real_image(image_bytes, onnx_session)
        if not is_real:
            logger.info(f"ONNX: Imagen rechazada - Tipo: {detected_type}, Confianza: {real_confidence:.4f}")
            return False, real_error, confianza_rostro, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": False
            }
        
        logger.info(f"ONNX: Imagen real validada - Confianza: {real_confidence:.4f} - Continuando a validaciones de calidad")

        # PASO 3: VALIDACIONES DE CALIDAD - FILTRO DE CALIDAD (Solo si es imagen real con 1 rostro)
        quality_valid, quality_error = validate_image_quality(image_bytes)
        if not quality_valid:
            logger.info(f"Calidad: Imagen rechazada - {quality_error}")
            return False, quality_error, confianza_rostro, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": True
            }
        
        logger.info("Calidad: Imagen aprobada - Todos los filtros pasados exitosamente")
        
        # TODOS LOS FILTROS PASADOS - IMAGEN VÁLIDA PARA CLASIFICACIÓN
        return True, "", confianza_rostro, {
            "tipo_detectado": detected_type,
            "confianza_tipo": real_confidence,
            "es_real": True
        }
    except Exception as e:
        logger.error(f"Error en validación de rostro: {e}")
        return False, f"Error al procesar la imagen para detección de rostros: {str(e)}", None, {
            "tipo_detectado": "error",
            "confianza_tipo": 0.0,
            "es_real": False
        }
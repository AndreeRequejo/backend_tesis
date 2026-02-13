"""
Servicio para validaciones de imagen y rostros
"""
import numpy as np
import io
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from utils.image_utils import preprocess_image_for_onnx
from config import *
from .blacklist import is_image_blacklisted
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)


def validate_real_image(image_bytes: bytes, onnx_session) -> tuple[bool, str]:
    """Validar que la imagen sea real usando ONNX"""
    try:
        input_array = preprocess_image_for_onnx(image_bytes, onnx_session)
        input_name = onnx_session.get_inputs()[0].name
        
        outputs = onnx_session.run(None, {input_name: input_array})
        probabilities = outputs[0][0]
        
        # Aplicar softmax si es necesario
        if np.sum(probabilities) > 1.01:
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        predicted_type = ANIME_CLASS_NAMES[predicted_class] if predicted_class < len(ANIME_CLASS_NAMES) else "unknown"
        
        logger.info(f"ONNX - Tipo: {predicted_type}, Confianza: {confidence:.4f}")
        
        if predicted_type == "real" and confidence >= REAL_CLASS_CONFIDENCE_THRESHOLD:
            return True, ""
        
        error_msg = FACE_VALIDATION_MESSAGES["low_real_confidence"] if predicted_type == "real" else FACE_VALIDATION_MESSAGES["anime_detected"]
        return False, error_msg
    
    except Exception as e:
        logger.error(f"Error validación ONNX: {e}")
        return False, f"Error al procesar: {str(e)}"


def validate_face_with_mtcnn(image_bytes: bytes, mtcnn_detector) -> tuple[bool, str]:
    """Validar rostros usando MTCNN"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        boxes, probs = mtcnn_detector.detect(image)
        
        if boxes is None or len(boxes) == 0:
            return False, FACE_VALIDATION_MESSAGES["no_face"]
        
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]
        
        if confidence < FACE_DETECTOR_CONFIG["mtcnn_confidence"]:
            logger.info(f"MTCNN: Confianza baja ({confidence:.3f})")
            return False, FACE_VALIDATION_MESSAGES["low_confidence"]
        
        logger.info(f"MTCNN: Rostro válido - Confianza: {confidence:.3f}")
        return True, ""
    except Exception as e:
        logger.error(f"Error MTCNN: {e}")
        return False, f"Error procesando imagen: {str(e)}"


def validate_face_in_image(image_bytes: bytes, onnx_session, mtcnn_detector) -> tuple[bool, str]:
    """Validar rostro único y que sea imagen real"""
    try:
        # PASO 0: Blacklist
        if is_image_blacklisted(image_bytes):
            logger.info("Imagen en blacklist")
            return False, FACE_VALIDATION_MESSAGES["no_face"]
        
        # Abrir y validar imagen
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            return False, "La imagen debe tener 3 canales RGB"

        # PASO 1: MediaPipe - Detección rápida (nueva API)
        # Configurar opciones del detector con el modelo de face_detection
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_FACE_DETECTION_MODEL)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=FACE_DETECTOR_CONFIG["mediapipe_confidence"]
        )
        
        # Crear detector y procesar imagen
        detector = vision.FaceDetector.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = detector.detect(mp_image)
        detector.close()
        
        if not detection_result.detections:
            return False, FACE_VALIDATION_MESSAGES["no_face"]
        
        if len(detection_result.detections) > 1:
            logger.info(f"MediaPipe: {len(detection_result.detections)} rostros")
            return False, FACE_VALIDATION_MESSAGES["multiple_faces"]

        # PASO 2: MTCNN - Validación precisa
        face_valid, face_error = validate_face_with_mtcnn(image_bytes, mtcnn_detector)
        if not face_valid:
            return False, face_error

        # PASO 3: ONNX - Validación de autenticidad
        is_real, real_error = validate_real_image(image_bytes, onnx_session)
        if not is_real:
            return False, real_error
        
        logger.info("✓ Validación completa exitosa")
        return True, ""
    except Exception as e:
        logger.error(f"Error validación: {e}")
        return False, f"Error procesando imagen: {str(e)}"
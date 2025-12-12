"""
Servicio para validaciones de imagen y rostros
"""
import numpy as np
import io
import logging
import mediapipe as mp
from PIL import Image
from utils.image_utils import preprocess_image_for_onnx
from config import *
from .blacklist import is_image_blacklisted

# Importación condicional de MTCNN
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet_pytorch no está disponible. Solo se podrá usar MediaPipe para detección de rostros.")

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


def validate_face_with_mtcnn(image_bytes: bytes, mtcnn_detector) -> tuple:
    """
    Validar rostros usando MTCNN de facenet_pytorch
    
    Args:
        image_bytes: Bytes de la imagen
        mtcnn_detector: Instancia de MTCNN inicializada
        
    Returns:
        tuple: (es_válida, mensaje_error, confianza_rostro, info_extra)
    """
    if not MTCNN_AVAILABLE:
        return False, "MTCNN no está disponible", None, {"detector": "unavailable"}
    
    if mtcnn_detector is None:
        return False, "Detector MTCNN no inicializado", None, {"detector": "not_initialized"}
    
    try:
        # Abrir imagen
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detectar rostros con MTCNN
        boxes, probs = mtcnn_detector.detect(image)
        
        # Si no se detectaron rostros (validación adicional después de MediaPipe)
        if boxes is None or len(boxes) == 0:
            logger.info("MTCNN: No se detectaron rostros")
            return False, FACE_VALIDATION_MESSAGES["no_face"], None, {
                "detector": "mtcnn",
                "faces_detected": 0,
                "confidence": 0.0
            }
        
        # Obtener el rostro con mayor confianza (MediaPipe ya verificó que hay 1 rostro)
        best_face_idx = np.argmax(probs)
        largest_box = boxes[best_face_idx]
        best_confidence = probs[best_face_idx]
        
        # Verificar confianza mínima del rostro detectado
        if best_confidence < FACE_DETECTOR_CONFIG["mtcnn_confidence"]:
            logger.info(f"MTCNN: Rostro con confianza baja ({best_confidence:.3f} < {FACE_DETECTOR_CONFIG['mtcnn_confidence']})")
            return False, FACE_VALIDATION_MESSAGES["low_confidence"], best_confidence, {
                "detector": "mtcnn",
                "faces_detected": len(boxes),
                "confidence": best_confidence,
                "threshold": FACE_DETECTOR_CONFIG["mtcnn_confidence"]
            }
        
        # Log de información de calidad
        logger.info(f"MTCNN: Rostro válido detectado - Confianza: {best_confidence:.3f}")
        
        return True, "", best_confidence, {
            "detector": "mtcnn",
            "faces_detected": 1,
            "confidence": best_confidence,
            "box": largest_box.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error en validación MTCNN: {e}")
        return False, f"Error al procesar la imagen con MTCNN: {str(e)}", None, {
            "detector": "mtcnn",
            "error": str(e)
        }


def validate_face_in_image(image_bytes: bytes, onnx_session, mtcnn_detector=None) -> tuple:
    """
    Validar que la imagen contenga exactamente un rostro y sea una imagen real
    
    FLUJO DE VALIDACIÓN:
    0. Blacklist: Verificar hash SHA-256 (instantáneo)
    1. MediaPipe: Detección RÁPIDA de múltiples rostros (filtro inicial ~50ms)
       → Si NO hay rostros o hay MÚLTIPLES rostros → RECHAZAR inmediatamente
    2. MTCNN: Validación PRECISA del rostro único (~100ms)
       → Verificar confianza del rostro
    3. ONNX Model: Validar si es imagen REAL vs ANIMADA (~100ms)
       → Si es ANIMADO/3D → RECHAZAR
    4. Continuar al modelo principal de clasificación de acné
    
    Args:
        image_bytes: Bytes de la imagen
        onnx_session: Sesión ONNX para validación
        mtcnn_detector: Instancia de MTCNN (requerido)
    Returns:
        tuple: (es_válida, mensaje_error, confianza_rostro, info_validacion)
    """
    try:
        # PASO 0: BLACKLIST - Verificar si la imagen está bloqueada (instantáneo)
        try:
            if is_image_blacklisted(image_bytes):
                logger.info("Imagen detectada en blacklist - rechazando")
                return False, FACE_VALIDATION_MESSAGES["no_face"], None, {
                    "tipo_detectado": "blacklist",
                    "confianza_tipo": 0.0,
                    "es_real": False
                }
        except Exception as e:
            logger.warning(f"Error verificando blacklist: {e}. Continuando validaciones.")

        # Convertir bytes a imagen RGB
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

        # PASO 1: MEDIAPIPE - Detección RÁPIDA de múltiples rostros (filtro inicial)
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=FACE_DETECTOR_CONFIG["mediapipe_confidence"]
        ) as face_detection:
            results = face_detection.process(image_np)
            
            # FILTRO: No hay rostros detectados
            if not results.detections or len(results.detections) == 0:
                logger.info("MediaPipe: No se detectaron rostros")
                return False, FACE_VALIDATION_MESSAGES["no_face"], None, {
                    "tipo_detectado": "no_face",
                    "confianza_tipo": 0.0,
                    "es_real": False,
                    "detector_info": {"detector": "mediapipe", "faces_detected": 0}
                }
            
            # FILTRO: Múltiples rostros detectados (rechazar inmediatamente)
            if len(results.detections) > 1:
                logger.info(f"MediaPipe: {len(results.detections)} rostros detectados - Rechazando")
                return False, FACE_VALIDATION_MESSAGES["multiple_faces"], None, {
                    "tipo_detectado": "multiple_faces",
                    "confianza_tipo": 0.0,
                    "es_real": False,
                    "detector_info": {"detector": "mediapipe", "faces_detected": len(results.detections)}
                }
            
            logger.info("MediaPipe: 1 rostro detectado - Continuando a validación MTCNN")

        # PASO 2: MTCNN - Validación PRECISA del rostro único
        if mtcnn_detector is None:
            logger.error("MTCNN no disponible para validación de rostro único")
            return False, "Detector de rostros no disponible", None, {
                "tipo_detectado": "error",
                "confianza_tipo": 0.0,
                "es_real": False
            }
        
        face_valid, face_error, confianza_rostro, detector_info = validate_face_with_mtcnn(image_bytes, mtcnn_detector)
        if not face_valid:
            logger.info(f"MTCNN: {face_error}")
            return False, face_error, confianza_rostro, {
                "tipo_detectado": "face_validation_failed",
                "confianza_tipo": 0.0,
                "es_real": False,
                "detector_info": detector_info
            }

        # PASO 3: ONNX MODEL - FILTRO DE AUTENTICIDAD (Solo si rostro único válido)
        is_real, real_confidence, detected_type, real_error = validate_real_image(image_bytes, onnx_session)
        if not is_real:
            logger.info(f"ONNX: Imagen rechazada - Tipo: {detected_type}, Confianza: {real_confidence:.4f}")
            return False, real_error, confianza_rostro, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": False,
                "detector_info": detector_info
            }
        
        logger.info("✓ Todos los filtros pasados exitosamente")
        
        # TODOS LOS FILTROS PASADOS - IMAGEN VÁLIDA PARA CLASIFICACIÓN
        return True, "", confianza_rostro, {
            "tipo_detectado": detected_type,
            "confianza_tipo": real_confidence,
            "es_real": True,
            "detector_info": detector_info
        }
    except Exception as e:
        logger.error(f"Error en validación de rostro: {e}")
        return False, f"Error al procesar la imagen para detección de rostros: {str(e)}", None, {
            "tipo_detectado": "error",
            "confianza_tipo": 0.0,
            "es_real": False,
            "detector_info": {"detector": "error", "error": str(e)}
        }
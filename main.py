import os
# Configurar TensorFlow para reducir mensajes informativos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageStat
import numpy as np
import cv2
import io
import logging
from datetime import datetime
import uvicorn

# Scalar documentation
from scalar_fastapi import get_scalar_api_reference, Layout

# Detección de rostros
import mediapipe as mp

# ONNX Runtime para modelo de anime/3D
import onnxruntime as ort

# Importaciones locales
from model import MyNet
from config import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# VARIABLES GLOBALES
# ===============================

model = None
device = None
onnx_session = None  # Sesión ONNX para detección de anime/3D

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ===============================
# MODELOS PYDANTIC
# ===============================

class PredictionResponse(BaseModel):
    """Respuesta de predicción individual"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "prediccion_class": 1,
                "prediccion_label": "Moderate Acne",
                "confianza": 0.8567,
                "confianza_porcentaje": "85.67%",
                "probabilidades": {
                    "Leve": 0.1234,
                    "Moderado": 0.8567,
                    "Severo": 0.0156
                },
                "validacion_imagen": {
                    "tipo_detectado": "real",
                    "confianza_tipo": 0.9834,
                    "es_real": True
                },
                "tiempo_procesamiento": 234.5
            }
        }
    )
    
    success: bool
    prediccion_class: int
    prediccion_label: str
    confianza: float
    confianza_porcentaje: str
    probabilidades: dict
    validacion_imagen: dict
    tiempo_procesamiento: float

class BatchResponse(BaseModel):
    """Respuesta de predicción en lote"""
    success: bool
    total_images: int
    successful: int
    failed: int
    predicciones: List[dict]
    tiempo_procesamiento: float

# ===============================
# FUNCIONES DE UTILIDAD
# ===============================

def load_model():
    """Cargar modelo entrenado y modelo ONNX"""
    global model, device, onnx_session

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar modelo de clasificación de acné
        try:
            model = torch.load(MODEL_PATH, weights_only=False, map_location=device)
            logger.info("Modelo de acné cargado")
        except:
            # Fallback: crear arquitectura y cargar pesos
            model = MyNet()
            checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=device)
            if hasattr(checkpoint, 'state_dict'):
                model.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            logger.info("Modelo de acné cargado")

        model = model.to(device)
        model.eval()

        # Cargar modelo ONNX para detección de anime/3D
        try:
            # Configurar proveedores ONNX (GPU si está disponible, sino CPU)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
            
            # Inspeccionar las dimensiones esperadas del modelo
            input_details = onnx_session.get_inputs()[0]
            output_details = onnx_session.get_outputs()[0]
            
            logger.info(f"Modelo ONNX cargado con proveedores: {onnx_session.get_providers()}")
            logger.info(f"Entrada esperada - Nombre: {input_details.name}, Forma: {input_details.shape}, Tipo: {input_details.type}")
            logger.info(f"Salida esperada - Nombre: {output_details.name}, Forma: {output_details.shape}, Tipo: {output_details.type}")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo ONNX: {e}")
            return False

        logger.info("Modelos listos")
        return True

    except Exception as e:
        logger.error(f"Error al cargar los modelos: {e}")
        return False

def process_image(image_bytes: bytes):
    """Procesar imagen para predicción"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict(image_tensor):
    """Realizar predicción"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
        
        # Crear diccionario de probabilidades
        probs_dict = {}
        probs_array = probabilities.cpu().numpy()[0]
        for i, prob in enumerate(probs_array):
            probs_dict[CLASS_NAMES[i]] = round(float(prob), 4)
        
        return predicted_class, confidence, probs_dict

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

def preprocess_image_for_onnx(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesar imagen para el modelo ONNX
    
    Args:
        image_bytes: Bytes de la imagen
        
    Returns:
        numpy array: Array de imagen preprocesado para ONNX
    """
    global onnx_session
    
    # Abrir y convertir imagen
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Obtener las dimensiones esperadas del modelo dinámicamente
    if onnx_session is not None:
        input_shape = onnx_session.get_inputs()[0].shape
        logger.info(f"Forma de entrada del modelo ONNX: {input_shape}")
        
        # Extraer dimensiones (asumiendo formato NCHW: [batch, channels, height, width])
        if len(input_shape) == 4:
            expected_height = input_shape[2] if isinstance(input_shape[2], int) else ONNX_IMAGE_SIZE
            expected_width = input_shape[3] if isinstance(input_shape[3], int) else ONNX_IMAGE_SIZE
        else:
            expected_height = expected_width = ONNX_IMAGE_SIZE
    else:
        expected_height = expected_width = ONNX_IMAGE_SIZE
    
    logger.info(f"Redimensionando imagen a: {expected_width}x{expected_height}")
    
    # Redimensionar a tamaño esperado por el modelo ONNX
    image = image.resize((expected_width, expected_height))
    
    # Convertir a array numpy y normalizar
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalizar a [0,1]
    
    # Reorganizar dimensiones para ONNX: (batch_size, channels, height, width)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    logger.info(f"Forma final del array: {image_array.shape}")
    
    return image_array

def validate_real_image(image_bytes: bytes) -> tuple:
    """
    Validar que la imagen sea una foto real de una persona usando modelo ONNX
    
    Args:
        image_bytes: Bytes de la imagen
        
    Returns:
        tuple: (es_real, confianza, tipo_detectado, mensaje_error)
    """
    global onnx_session
    
    if onnx_session is None:
        return False, 0.0, "unknown", "Modelo ONNX no disponible"
    
    try:
        # Preprocesar imagen
        input_array = preprocess_image_for_onnx(image_bytes)
        
        # Obtener información de entrada del modelo ONNX
        input_name = onnx_session.get_inputs()[0].name
        expected_shape = onnx_session.get_inputs()[0].shape
        
        logger.info(f"Entrada del modelo - Nombre: {input_name}, Forma esperada: {expected_shape}, Forma actual: {input_array.shape}")
        
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

def validate_face_in_image(image_bytes: bytes) -> tuple:
    """
    Validar que la imagen contenga exactamente un rostro y sea una imagen real
    Args:
        image_bytes: Bytes de la imagen
    Returns:
        tuple: (es_válida, mensaje_error, confianza_rostro, info_validacion)
    """
    try:
        # 1. Validar que sea una imagen real usando modelo ONNX
        is_real, real_confidence, detected_type, real_error = validate_real_image(image_bytes)
        if not is_real:
            return False, real_error, None, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": False
            }

        # 2. Validar calidad general de la imagen
        quality_valid, quality_error = validate_image_quality(image_bytes)
        if not quality_valid:
            return False, quality_error, None, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": True
            }

        # 3. Convertir los bytes a imagen RGB
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"No se pudo abrir la imagen: {e}")
            return False, "No se pudo abrir la imagen. Asegúrate de que el archivo es una imagen válida.", None, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": True
            }

        image_np = np.array(image)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            logger.error("La imagen no tiene 3 canales RGB")
            return False, "La imagen debe tener 3 canales (RGB).", None, {
                "tipo_detectado": detected_type,
                "confianza_tipo": real_confidence,
                "es_real": True
            }

        # 4. Usar mediapipe para detección de rostros
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8) as face_detection:
            results = face_detection.process(image_np)
            if not results.detections or len(results.detections) == 0:
                return False, FACE_VALIDATION_MESSAGES["no_face"], None, {
                    "tipo_detectado": detected_type,
                    "confianza_tipo": real_confidence,
                    "es_real": True
                }
            if len(results.detections) > 1:
                return False, FACE_VALIDATION_MESSAGES["multiple_faces"], None, {
                    "tipo_detectado": detected_type,
                    "confianza_tipo": real_confidence,
                    "es_real": True
                }
            
            # Obtener confianza del único rostro
            confianza_rostro = results.detections[0].score[0] if results.detections[0].score else None
            
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

# ===============================
# LIFESPAN
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando servidor...")
    success = load_model()
    if success:
        logger.info("Servidor listo")
    else:
        logger.error("Error al iniciar")
    
    yield
    
    # Shutdown
    logger.info("Cerrando servidor...")

# ===============================
# FASTAPI APP
# ===============================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_url="/openapi.json",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# DOCUMENTACIÓN SCALAR
# ===============================

@app.get("/", include_in_schema=False)
async def scalar_docs():
    """Documentación con Scalar"""
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        layout=Layout.MODERN,
        show_sidebar=True,
        default_open_all_tags=True,
        hide_download_button=True,
        hide_models=False,
    )

# ===============================
# ENDPOINTS
# ===============================

@app.get("/status", tags=["General"])
async def server():
    """Información de la API"""
    return {
        "message": "Documentación Tesis Acne",
        "version": API_VERSION,
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "batch": "/predict/batch"
        },
        "model_ready": model is not None,
        "onnx_model_ready": onnx_session is not None,
        "face_detection_ready": True,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    file: UploadFile = File(..., description="Imagen para clasificar"),
    skip_quality_check: bool = False
):
    start_time = datetime.utcnow()
    try:
        # Validaciones
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo no disponible"
            )

        if not validate_image(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de archivo no válido: {file.content_type}"
            )

        # Procesar imagen
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Archivo vacío"
            )

        # Validar que la imagen contenga un rostro
        # Si skip_quality_check es True, temporalmente deshabilitar validaciones de calidad
        original_quality_setting = IMAGE_QUALITY_CONFIG.get("enable_quality_checks", True)
        if skip_quality_check:
            IMAGE_QUALITY_CONFIG["enable_quality_checks"] = False

        try:
            face_valid, face_error_msg, face_confidence, validation_info = validate_face_in_image(image_bytes)
            if face_confidence is not None:
                try:
                    logger.info(f"Confianza del rostro detectado: {face_confidence:.4f}")
                except Exception:
                    logger.info(f"Confianza del rostro detectado: {face_confidence}")
            
            # Loggear información de validación ONNX
            if validation_info:
                logger.info(f"Validación ONNX - Tipo: {validation_info.get('tipo_detectado', 'unknown')}, "
                          f"Confianza: {validation_info.get('confianza_tipo', 0.0):.4f}, "
                          f"Es real: {validation_info.get('es_real', False)}")
            
            if not face_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=face_error_msg
                )
        finally:
            # Restaurar configuración original
            IMAGE_QUALITY_CONFIG["enable_quality_checks"] = original_quality_setting

        image_tensor = process_image(image_bytes)

        # Predecir
        predicted_class, confidence, probabilities = predict(image_tensor)

        # Tiempo de procesamiento
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return PredictionResponse(
            success=True,
            prediccion_class=predicted_class,
            prediccion_label=CLASS_NAMES[predicted_class],
            confianza=round(confidence, 4),
            confianza_porcentaje=f"{confidence*100:.2f}%",
            probabilidades=probabilities,
            validacion_imagen=validation_info or {
                "tipo_detectado": "unknown",
                "confianza_tipo": 0.0,
                "es_real": False
            },
            tiempo_procesamiento=round(processing_time, 2)
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Lista de imágenes (máx 3)")
):
    """
    Clasificar múltiples imágenes
    
    Sube hasta 3 imágenes y obtén las clasificaciones.
    Cada imagen debe contener al menos un rostro visible.
    """
    start_time = datetime.utcnow()
    try:
        # Validaciones
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo no disponible"
            )

        if len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se enviaron archivos"
            )

        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Máximo {MAX_BATCH_SIZE} archivos"
            )

        # Procesar cada archivo
        predictions = []
        successful = 0

        for file in files:
            try:
                if not validate_image(file):
                    predictions.append({
                        "filename": file.filename,
                        "success": False,
                        "error": f"Tipo no válido: {file.content_type}"
                    })
                    continue

                image_bytes = await file.read()

                # Validar rostro en la imagen
                face_valid, face_error_msg, face_confidence, validation_info = validate_face_in_image(image_bytes)
                if face_confidence is not None:
                    try:
                        logger.info(f"Confianza del rostro detectado en '{file.filename}': {face_confidence:.4f}")
                    except Exception:
                        logger.info(f"Confianza del rostro detectado en '{file.filename}': {face_confidence}")
                
                # Loggear información de validación ONNX
                if validation_info:
                    logger.info(f"Validación ONNX para '{file.filename}' - Tipo: {validation_info.get('tipo_detectado', 'unknown')}, "
                              f"Confianza: {validation_info.get('confianza_tipo', 0.0):.4f}, "
                              f"Es real: {validation_info.get('es_real', False)}")
                
                if not face_valid:
                    predictions.append({
                        "filename": file.filename,
                        "success": False,
                        "error": face_error_msg
                    })
                    continue

                image_tensor = process_image(image_bytes)
                predicted_class, confidence, probabilities = predict(image_tensor)

                predictions.append({
                    "filename": file.filename,
                    "success": True,
                    "prediccion_class": predicted_class,
                    "prediccion_label": CLASS_NAMES[predicted_class],
                    "confianza": round(confidence, 4),
                    "confianza_porcentaje": f"{confidence*100:.2f}%",
                    "probabilidades": probabilities,
                    "validacion_imagen": validation_info or {
                        "tipo_detectado": "unknown",
                        "confianza_tipo": 0.0,
                        "es_real": False
                    }
                })

                successful += 1

            except Exception as e:
                predictions.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return BatchResponse(
            success=True,
            total_images=len(files),
            successful=successful,
            failed=len(files) - successful,
            predicciones=predictions,
            tiempo_procesamiento=round(processing_time, 2)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

# ===============================
# EJECUTAR
# ===============================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True
    )
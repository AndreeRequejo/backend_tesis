import os
# Configurar TensorFlow para reducir mensajes informativos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Cambiar a 2 para suprimir warnings
os.environ['TF_LITE_DISABLE_FEEDBACK_TENSOR'] = '1'  # Suprimir mensajes de feedback tensor
os.environ['GLOG_minloglevel'] = '2'  # Suprimir logs de Google (MediaPipe)

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import uvicorn

# Scalar documentation
from scalar_fastapi import get_scalar_api_reference, Layout

# Importaciones locales
from config import *
from models.schemas import PredictionResponse, BatchResponse
from services.model_service import model_service
from services.validation_service import validate_face_in_image
from utils.image_utils import validate_image, process_image

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# LIFESPAN
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando servidor...")
    success = model_service.load_models()
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
    detector_info = model_service.get_detector_info()
    
    return {
        "message": "Documentación Tesis Acne",
        "version": API_VERSION,
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "batch": "/predict/batch"
        },
        "models_status": {
            "acne_model": model_service.is_model_ready(),
            "onnx_model": model_service.is_onnx_ready(),
            "face_detector": detector_info
        },
        "face_detection": {
            "current_detector": detector_info["detector"],
            "mtcnn_available": detector_info.get("available", False) if detector_info["detector"] == "mtcnn" else False,
            "change_instructions": "Modifica FACE_DETECTOR_CONFIG['use_mtcnn'] en config.py para cambiar entre MediaPipe y MTCNN"
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    file: UploadFile = File(..., description="Imagen para clasificar"),
    skip_quality_check: bool = False
):
    start_time = datetime.utcnow()
    try:
        # Validaciones
        if not model_service.is_model_ready():
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

        # FLUJO DE VALIDACIÓN:
        # 1. MediaPipe: Detectar presencia y cantidad de rostros (filtro rápido inicial)
        # 2. ONNX Model: Validar si es imagen real vs animada (solo si hay 1 rostro)  
        # 3. Validaciones de calidad: Blur, contraste, brillo (solo si es imagen real)
        # 4. Proceder al modelo principal de clasificación de acné
        
        # Si skip_quality_check es True, temporalmente deshabilitar validaciones de calidad
        original_quality_setting = IMAGE_QUALITY_CONFIG.get("enable_quality_checks", True)
        if skip_quality_check:
            IMAGE_QUALITY_CONFIG["enable_quality_checks"] = False

        try:
            face_valid, face_error_msg, face_confidence, validation_info = validate_face_in_image(
                image_bytes, 
                model_service.get_onnx_session(),
                model_service.get_mtcnn()  # Pasar el detector MTCNN
            )
            
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
        predicted_class, confidence, probabilities = model_service.predict(image_tensor, temperature=0.4)

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
        if not model_service.is_model_ready():
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

                # FLUJO DE VALIDACIÓN: MediaPipe/MTCNN → ONNX → Calidad → Modelo Principal
                face_valid, face_error_msg, face_confidence, validation_info = validate_face_in_image(
                    image_bytes, 
                    model_service.get_onnx_session(),
                    model_service.get_mtcnn()  # Pasar el detector MTCNN
                )
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
                predicted_class, confidence, probabilities = model_service.predict(image_tensor)

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
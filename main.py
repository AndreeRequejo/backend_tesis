from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import logging
from datetime import datetime
import uvicorn

# Scalar documentation
from scalar_fastapi import get_scalar_api_reference, Layout

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
                    "Severo": 0.0156,
                    "Muy Severo": 0.0043
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
    """Cargar modelo entrenado"""
    global model, device
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Intentar cargar el modelo
        try:
            model = torch.load(MODEL_PATH, weights_only=False, map_location=device)
            logger.info("Modelo cargado")
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
            logger.info("Modelo cargado")
        
        model = model.to(device)
        model.eval()
        
        logger.info("Modelo listo")
        return True
        
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
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

@app.get("/docs", include_in_schema=False)
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

@app.get("/", tags=["General"])
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
        "model_ready": model is not None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    file: UploadFile = File(..., description="Imagen para clasificar")
):
    """
    Clasificar una sola imagen
    
    Sube una imagen y obtén la clasificación de severidad del acné
    """
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
    files: List[UploadFile] = File(..., description="Lista de imágenes (máx 10)")
):
    """
    Clasificar múltiples imágenes
    
    Sube hasta 3 imágenes y obtén las clasificaciones
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
                image_tensor = process_image(image_bytes)
                predicted_class, confidence, probabilities = predict(image_tensor)
                
                predictions.append({
                    "filename": file.filename,
                    "success": True,
                    "prediccion_class": predicted_class,
                    "prediccion_label": CLASS_NAMES[predicted_class],
                    "confianza": round(confidence, 4),
                    "confianza_porcentaje": f"{confidence*100:.2f}%",
                    "probabilidades": probabilities
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
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import uvicorn
from scalar_fastapi import get_scalar_api_reference, Layout
from config import *
from models.schemas import PredictionResponse, BatchResponse
from services.model_service import model_service
from utils.helpers import check_model_ready, validate_and_read_image, process_single_prediction

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# LIFESPAN
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Iniciando
    logger.info("Iniciando servidor...")
    success = model_service.load_models()
    if success:
        logger.info("Servidor listo")
    else:
        logger.error("Error al iniciar")
    
    yield
    # Apagando
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
    return {
        "message": "Documentación Tesis Acne",
        "version": API_VERSION,
        "status": "running"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(file: UploadFile = File(..., description="Imagen para clasificar")):
    start_time = datetime.utcnow()
    try:
        check_model_ready()
        image_bytes = await validate_and_read_image(file)
        result = process_single_prediction(image_bytes, temperature=0.44)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            success=True,
            tiempo_procesamiento=round(processing_time, 2),
            **result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(..., description="Lista de imágenes (máx 3)")):
    """Clasificar múltiples imágenes. Sube hasta 3 imágenes y obtén las clasificaciones."""
    start_time = datetime.utcnow()
    try:
        check_model_ready()
        
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se enviaron archivos"
            )

        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Máximo {MAX_BATCH_SIZE} archivos"
            )

        predictions = []
        successful = 0

        for file in files:
            try:
                image_bytes = await validate_and_read_image(file)
                result = process_single_prediction(image_bytes, temperature=0.4)
                
                predictions.append({
                    "filename": file.filename,
                    "success": True,
                    **result
                })
                successful += 1

            except HTTPException as e:
                predictions.append({
                    "filename": file.filename,
                    "success": False,
                    "error": e.detail
                })
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
        reload=False
    )
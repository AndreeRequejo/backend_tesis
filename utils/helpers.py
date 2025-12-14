"""
Funciones helper para endpoints
"""
from fastapi import UploadFile, HTTPException, status
from services.model_service import model_service
from services.validation_service import validate_face_in_image
from utils.image_utils import validate_image, process_image
from config import CLASS_NAMES


def check_model_ready():
    """Verificar que el modelo esté listo"""
    if not model_service.is_model_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )


async def validate_and_read_image(file: UploadFile) -> bytes:
    """Validar tipo y leer imagen"""
    if not validate_image(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no válido: {file.content_type}"
        )
    
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Archivo vacío"
        )
    
    return image_bytes


def process_single_prediction(image_bytes: bytes, temperature: float = 0.44) -> dict:
    """Procesar una predicción completa"""
    # Validar rostro
    face_valid, face_error_msg = validate_face_in_image(
        image_bytes, 
        model_service.onnx_session,
        model_service.mtcnn
    )
    
    if not face_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=face_error_msg
        )

    # Procesar y predecir
    image_tensor = process_image(image_bytes)
    predicted_class, confidence, probabilities = model_service.predict(image_tensor, temperature)

    return {
        "prediccion_class": predicted_class,
        "prediccion_label": CLASS_NAMES[predicted_class],
        "confianza": round(confidence, 4),
        "confianza_porcentaje": f"{confidence*100:.2f}%",
        "probabilidades": probabilities,
    }

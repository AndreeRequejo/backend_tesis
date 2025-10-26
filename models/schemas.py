"""
Esquemas Pydantic para la API de clasificación de acné
"""
from pydantic import BaseModel, ConfigDict
from typing import List


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
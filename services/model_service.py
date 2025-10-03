"""
Servicio para la carga y uso de modelos ML
"""
import torch
import torch.nn.functional as F
import onnxruntime as ort
import logging
from model import MyNet
from config import *

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.device = None
        self.onnx_session = None

    def load_models(self):
        """Cargar modelo entrenado y modelo ONNX"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Cargar modelo de clasificación de acné
            try:
                self.model = torch.load(MODEL_PATH, weights_only=False, map_location=self.device)
                logger.info("Modelo de acné cargado")
            except:
                # Fallback: crear arquitectura y cargar pesos
                self.model = MyNet()
                checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=self.device)
                if hasattr(checkpoint, 'state_dict'):
                    self.model.load_state_dict(checkpoint.state_dict())
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model = checkpoint
                logger.info("Modelo de acné cargado")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Cargar modelo ONNX para detección de anime/3D
            try:
                # Verificar proveedores disponibles y usar GPU si está disponible
                available_providers = ort.get_available_providers()
                
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    logger.info("ONNX CUDAExecutionProvider")
                else:
                    providers = ['CPUExecutionProvider']
                    logger.info("ONNX CPUExecutionProvider")
                
                self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)    
                
            except Exception as e:
                logger.error(f"Error al cargar modelo ONNX: {e}")
                return False
            
            return True

        except Exception as e:
            logger.error(f"Error al cargar los modelos: {e}")
            return False

    def predict(self, image_tensor):
        """Realizar predicción"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
            
            # Crear diccionario de probabilidades
            probs_dict = {}
            probs_array = probabilities.cpu().numpy()[0]
            for i, prob in enumerate(probs_array):
                probs_dict[CLASS_NAMES[i]] = round(float(prob), 4)
            
            return predicted_class, confidence, probs_dict

    def is_model_ready(self):
        """Verificar si el modelo está listo"""
        return self.model is not None

    def is_onnx_ready(self):
        """Verificar si el modelo ONNX está listo"""
        return self.onnx_session is not None

    def get_onnx_session(self):
        """Obtener sesión ONNX"""
        return self.onnx_session


# Instancia global del servicio
model_service = ModelService()
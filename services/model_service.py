"""
Servicio para la carga y uso de modelos ML
"""
import torch
import torch.nn.functional as F
import onnxruntime as ort
import logging
from utils.model import MyNet
from config import *
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.device = None
        self.onnx_session = None
        self.mtcnn = None

    def _get_onnx_providers(self):
        """Determinar proveedores ONNX (CUDA o CPU)"""
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available:
            return ['CPUExecutionProvider']
        
        try:
            # Probar CUDA
            ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except Exception as e:
            logger.warning(f"CUDA no disponible para ONNX: {e}")
            return ['CPUExecutionProvider']

    def load_models(self):
        """Cargar todos los modelos necesarios"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Cargar modelo de acné
            self.model = MyNet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("Modelo de acné cargado")
            
            # Cargar modelo ONNX
            providers = self._get_onnx_providers()
            self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
            logger.info(f"ONNX cargado - {providers[0]}")
            
            # Inicializar MTCNN
            self.mtcnn = MTCNN(
                image_size=MTCNN_CONFIG["image_size"],
                margin=MTCNN_CONFIG["margin"],
                device=self.device
            )
            logger.info("MTCNN inicializado")
            
            return True
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            return False

    def predict(self, image_tensor, temperature: float = 1.0):
        """Realizar predicción con softmax ajustable por temperatura"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)

            # Aplicar softmax con temperatura
            probabilities = F.softmax(outputs / temperature, dim=1)

            # Clase con mayor probabilidad
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

            # Crear diccionario de probabilidades
            probs_dict = {}
            probs_array = probabilities.cpu().numpy()[0]
            for i, prob in enumerate(probs_array):
                probs_dict[CLASS_NAMES[i]] = round(float(prob), 4)

            return predicted_class, confidence, probs_dict

    def is_model_ready(self):
        """Verificar si todos los modelos están listos"""
        return all([self.model, self.onnx_session, self.mtcnn])


# Instancia global del servicio
model_service = ModelService()
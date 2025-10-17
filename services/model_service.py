"""
Servicio para la carga y uso de modelos ML
"""
import torch
import torch.nn.functional as F
import onnxruntime as ort
import logging
from model import MyNet
from config import *

# Importación condicional de MTCNN
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet_pytorch no está disponible. Solo se podrá usar MediaPipe para detección de rostros.")

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.device = None
        self.onnx_session = None
        self.mtcnn = None

    def load_models(self):
        """Cargar modelo entrenado y modelo ONNX"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Cargar modelo de clasificación de acné
            try:
                self.model = MyNet()
                self.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=self.device))
                logger.info("Modelo de acné cargado")
            except:
                # Fallback: crear arquitectura y cargar pesos
                self.model = MyNet()
                self.model = torch.load(MODEL_PATH, weights_only=False, map_location=self.device)
                logger.info("Modelo de acné cargado")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Cargar modelo ONNX para detección de anime/3D
            try:
                # Verificar proveedores disponibles y usar GPU si está disponible
                available_providers = ort.get_available_providers()
                
                # Intentar primero con CUDA, pero usar CPU como fallback
                providers = ['CPUExecutionProvider']  # Usar CPU por defecto
                
                if 'CUDAExecutionProvider' in available_providers:
                    try:
                        # Probar si CUDA realmente funciona
                        test_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        test_session = None  # Liberar memoria
                    except Exception as cuda_error:
                        logger.warning(f"ONNX: CUDA no funcional, usando CPU. Error: {cuda_error}")
                        providers = ['CPUExecutionProvider']
                
                # Crear la sesión final con los proveedores seleccionados
                self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
                logger.info(f"ONNX: Modelo cargado - {providers}")
                
            except Exception as e:
                logger.error(f"Error al cargar modelo ONNX: {e}")
                return False

            # Inicializar MTCNN si está configurado y disponible
            if FACE_DETECTOR_CONFIG["use_mtcnn"] and MTCNN_AVAILABLE:
                try:
                    self.mtcnn = MTCNN(
                        image_size=MTCNN_CONFIG["image_size"],
                        margin=MTCNN_CONFIG["margin"],
                        min_face_size=MTCNN_CONFIG["min_face_size"],
                        thresholds=MTCNN_CONFIG["thresholds"],
                        factor=MTCNN_CONFIG["factor"],
                        post_process=MTCNN_CONFIG["post_process"],
                        device=self.device,
                        keep_all=MTCNN_CONFIG["keep_all"],
                        select_largest=MTCNN_CONFIG["select_largest"]
                    )
                    logger.info("MTCNN inicializado correctamente")
                except Exception as e:
                    logger.error(f"Error al inicializar MTCNN: {e}")
                    return False
            elif FACE_DETECTOR_CONFIG["use_mtcnn"] and not MTCNN_AVAILABLE:
                logger.warning("facenet_pytorch no está disponible")
            else:
                logger.info("Usando MediaPipe para detección de rostros")
            
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

    def is_mtcnn_ready(self):
        """Verificar si MTCNN está listo"""
        return self.mtcnn is not None

    def get_mtcnn(self):
        """Obtener detector MTCNN"""
        return self.mtcnn

    def get_detector_info(self):
        """Obtener información sobre el detector configurado"""
        if FACE_DETECTOR_CONFIG["use_mtcnn"]:
            return {
                "detector": "mtcnn",
                "available": MTCNN_AVAILABLE,
                "ready": self.is_mtcnn_ready(),
                "config": MTCNN_CONFIG
            }
        else:
            return {
                "detector": "mediapipe",
                "available": True,  # MediaPipe siempre está disponible
                "ready": True,  # No requiere inicialización especial
                "config": {"confidence": FACE_DETECTOR_CONFIG["mediapipe_confidence"]}
            }


# Instancia global del servicio
model_service = ModelService()
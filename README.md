# API de Clasificación de Severidad de Acné

Esta API permite clasificar la **severidad del acné** en imágenes faciales mediante un modelo de **Deep Learning** entrenado con **EfficientNetV2** y desplegado con **FastAPI**.

## 🚀 Características

* Clasificación de severidad en 4 categorías:

  * **0**: Acné Leve
  * **1**: Acné Moderado
  * **2**: Acné Severo
  
* Predicción **individual** de imágenes.
* Predicción **en lote** (hasta 3 imágenes a la vez).
* Soporte para imágenes en formatos: **JPG, JPEG, PNG**.
* Documentación interactiva con **Scalar** (`/docs`).

---

## 📂 Estructura del Proyecto

```
.
├── main.py              # Código principal de la API (FastAPI)
├── model.py             # Definición de la arquitectura del modelo
├── config.py            # Configuración de constantes y parámetros
├── acne_model.pt        # Pesos del modelo entrenado (no incluido en repo público)
├── requirements.txt     # Dependencias del proyecto
└── README.md            # Documentación
```

---

## ⚙️ Instalación y Configuración

1. **Clonar el repositorio**

2. **Crear entorno virtual e instalar dependencias**

```bash
pip install -r requirements.txt
```

3. **Colocar el modelo entrenado**

   * Ubica tu archivo `acne_model.pt` en la raíz del proyecto.

4. **Ejecutar el servidor**

```bash
python main.py
```

El servidor se iniciará en:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📡 Endpoints

### 1. **Información General**

```http
GET /
```

Retorna información básica de la API.

---

### 2. **Predicción Individual**

```http
POST /predict
```

**Parámetro**:

* `file`: imagen en formato **JPEG, JPG, PNG**

**Ejemplo con cURL**:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@ejemplo.jpg"
```

**Respuesta**:

```json
{
  "success": true,
  "prediccion_class": 1,
  "prediccion_label": "Moderado",
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
```

---

### 3. **Predicción en Lote**

```http
POST /predict/batch
```

**Parámetro**:

* `files`: lista de imágenes (máx. 3 archivos)

**Ejemplo con cURL**:

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

**Respuesta (ejemplo)**:

```json
{
  "success": true,
  "total_images": 2,
  "successful": 2,
  "failed": 0,
  "predicciones": [
    {
      "filename": "img1.jpg",
      "success": true,
      "prediccion_class": 0,
      "prediccion_label": "Leve",
      "confianza": 0.7234,
      "confianza_porcentaje": "72.34%",
      "probabilidades": {
        "Leve": 0.7234,
        "Moderado": 0.2012,
        "Severo": 0.0567,
        "Muy Severo": 0.0187
      }
    },
    {
      "filename": "img2.jpg",
      "success": true,
      "prediccion_class": 2,
      "prediccion_label": "Severo",
      "confianza": 0.8012,
      "confianza_porcentaje": "80.12%",
      "probabilidades": {
        "Leve": 0.1023,
        "Moderado": 0.0765,
        "Severo": 0.8012,
        "Muy Severo": 0.0200
      }
    }
  ],
  "tiempo_procesamiento": 452.7
}
```

---

### 4. **Documentación**

```http
GET /docs
```

Interfaz visual generada con **Scalar**.

---

## 🧠 Modelo

El modelo base es **EfficientNetV2-M** preentrenado en ImageNet, adaptado a la clasificación de 4 clases mediante un clasificador totalmente conectado con varias capas intermedias y regularización (`Dropout`).

Entrenado con:

* **PyTorch**
* **Label Smoothing Loss**
* **Optimizador Adam**

---

## 📌 Requisitos

Archivo `requirements.txt` mínimo:

```
fastapi
uvicorn
torch
torchvision
pillow
numpy
scalar-fastapi
```

---

## 👤 Autor

**Andree Requejo Díaz**.

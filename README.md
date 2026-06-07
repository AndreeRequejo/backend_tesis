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

## ⚙️ Instalación y Configuración

1. **Clonar el repositorio**

2. **Crear entorno virtual e instalar dependencias**

```bash
  uv venv
  uv sync
```

3. **Colocar el modelo entrenado**

   * Ubica tu archivo `acne_model.pt` en la carpeta core.

4. **Ejecutar el servidor**

```bash
  uv run uvicorn app.main:app --reload
  uv run uvicorn main:app
```

El servidor se iniciará en:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 🧠 Modelo

El modelo base es **EfficientNetV2-M** preentrenado en ImageNet, adaptado a la clasificación de 3 clases mediante un clasificador totalmente conectado con varias capas intermedias y regularización (`Dropout`).

Entrenado con:

* **PyTorch**
* **Label Smoothing Loss**
* **Optimizador Adam**

## 👤 Autor

**Andree Requejo Díaz**.

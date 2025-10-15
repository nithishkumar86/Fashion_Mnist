# 🧥 Fashion MNIST Classification using CNN and FastAPI

This project demonstrates a **Deep Learning** workflow using **Convolutional Neural Networks (CNNs)** to classify images from the **Fashion MNIST** dataset.  
It also includes a **FastAPI backend** for model inference and an optional **Streamlit frontend** for user interaction.
### first clone the repository 
```python

```
### create virtual environment and activate it 
```
python -m venv my venv

venv\Scripts\activate 
```

### install Dependencies 
```python
pip install -r requirements.txt
```

---

## 📦 1. Dataset Loading and Preprocessing

### Importing and Loading Dataset
```python
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

### Data Splitting and Reshaping
Fashion MNIST images are **28x28 grayscale**. We reshape them to include a channel dimension for CNN input.
```python
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

### Normalization
We normalize pixel values to the range `[0, 1]` for faster convergence.
```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

### One-Hot Encoding Labels
```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

---

## 🧠 2. Building the CNN Model

We use a **Sequential** CNN model with multiple convolutional and pooling layers.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

## ⚙️ 3. Model Compilation and Training

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

You can visualize accuracy/loss using:
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()
```

---

## 💾 4. Model Saving and Loading

After training, save the model for later inference:
```python
model.save('my_fashion_mnist_model.keras')
```

To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model('my_fashion_mnist_model.keras')
```

---

## 🌐 5. FastAPI Backend for Prediction

### Backend Code (`backend_server.py`)
```python
from fastapi import FastAPI, Body
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = load_model('D:\Fashion_mnist\model\my_fashion_mnist_model.keras')

class PredictionRequest(BaseModel):
    instances: list

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Backend Server!"}

@app.post("/prediction")
def predict(data: PredictionRequest = Body(...)):
    try:
        instances = np.array(data.instances)
        predictions = model.predict(instances)[0]
        predicted_class = int(np.argmax(predictions))
        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}
```

Run the backend:
```bash
uvicorn backend_server:app --host 0.0.0.0 --port 8000
```

---

## 💻 6. Streamlit Frontend for Image Upload

### Frontend Code (`streamlits.py`)
```python
import streamlit as st
import requests
import numpy as np
from PIL import Image

st.title("👕 Fashion MNIST Classifier")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

uploaded_file = st.file_uploader("Upload a Fashion MNIST-like image", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/prediction", json={"instances": img_array.tolist()})
        if response.status_code == 200:
            predicted_class = response.json().get("predicted_class")
            st.success(f"Predicted Class: {class_names[predicted_class]}")
        else:
            st.error("Prediction request failed.")
```

Run Streamlit:
```bash
streamlit run streamlits.py
```

---

## 🏁 Conclusion
This project demonstrates a complete **Deep Learning pipeline**:
- Data preprocessing  
- CNN model training  
- Backend inference using FastAPI  
- Frontend visualization with Streamlit  

You can extend it by:
- Adding more layers or tuning hyperparameters  
- Deploying to the cloud (AWS, Render, etc.)  
- Improving UI with image previews and confidence scores  


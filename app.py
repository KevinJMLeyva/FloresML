# Crea un ambiente virtual con python 3.10
# Instalar las librerias y correr 
# .\venv\Scripts\Activate.ps1   
# Dentro de la virtual machine correr  streamlit run app.py  
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model("mi_modelo (1).keras")

# Clases (ajusta según tu dataset)
class_names = [
 'black_eyed_susan', 'calendula', 'california_poppy', 
 'common_daisy', 'coreopsis', 'daffodil', 'dandelion', 
 'iris', 'magnolia', 'rose', 'sunflower', 'tulip', 'water_lily'
]

st.title("Clasificador de imágenes VGG11🌸")

st.text("Este modelo idenfica las siguientes flores: Black eyed susan, Calendula, California poppy, Common daisy, Coreopsis, Daffodil, Dandelion, Iris, Magnolia, Rose, Sunflower, Tulip, Water lily")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento
    img = image.resize((120,120))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = model.predict(img)[0]

    # Índice de mayor probabilidad
    top_idx = np.argmax(pred)
    clase = class_names[top_idx]
    confianza = pred[top_idx]

    st.write(f"### 🌸 Predicción: {clase}")
    st.write(f"Confianza: {confianza:.2f}")

    st.subheader("📊 Probabilidades por clase")

    # Crear diccionario clase -> probabilidad
    probs = {class_names[i]: float(pred[i]) for i in range(len(class_names))}

    # Ordenar de mayor a menor
    probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    # Mostrar barras
    for clase, prob in probs_sorted.items():
        st.write(clase)
        st.progress(prob)
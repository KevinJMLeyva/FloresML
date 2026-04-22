# Crea un ambiente virtual con python 3.10
# Instalar las librerias y correr 
# .\venv\Scripts\Activate.ps1   
# Dentro de la virtual machine correr  streamlit run app2.py  
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mi_modelo2 (2).keras", compile=False)
    return model

model = load_model()


class_names = [
 'black_eyed_susan', 'calendula', 'california_poppy', 
 'common_daisy', 'coreopsis', 'daffodil', 'dandelion', 
 'iris', 'magnolia', 'rose', 'sunflower', 'tulip', 'water_lily'
]


st.title("🌸 Clasificador de Flores con VGG16")
st.text("Este modelo idenfica las siguientes flores: Black eyed susan, Calendula, California poppy, Common daisy, Coreopsis, Daffodil, Dandelion, Iris, Magnolia, Rose, Sunflower, Tulip, Water lily")

st.write("Sube una imagen y el modelo predecirá la flor")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento (VGG16)
    img = image.resize((120, 120))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)[0]

    # Clase más probable
    top_idx = np.argmax(prediction)
    predicted_class = class_names[top_idx]
    confidence = prediction[top_idx]

    # Resultado
    st.subheader("Resultado:")
    st.write(f"🌼 Clase: **{predicted_class}**")
    st.write(f"📊 Confianza: **{confidence:.2f}**")


    st.subheader("📊 Probabilidades por clase")

    # Diccionario clase -> probabilidad
    probs = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}

    # Ordenar de mayor a menor
    probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    # Mostrar barras
    for clase, prob in probs_sorted.items():
        st.write(clase)
        st.progress(prob)
    st.subheader("Resultado:")
    st.write(f"🌼 Clase: **{predicted_class}**")
    st.write(f"📊 Confianza: **{confidence:.2f}**")

    # Mostrar probabilidades
    st.subheader("Probabilidades por clase:")
    for i, prob in enumerate(prediction):
        st.write(f"{class_names[i]}: {prob:.4f}")
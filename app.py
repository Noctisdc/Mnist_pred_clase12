import os
import random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Reto de caligrafía", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gochi+Hand&display=swap');
.stApp {
    background-color: #1f3a2e;
}
[data-testid="stHeader"] {
    background: transparent;
}
.stApp p, .stApp label, .stApp li {
    color: #f1f1e6;
}
button[data-baseweb="tab"] p {
    color: #f1f1e6 !important;
}
h1, h2, h3 {
    font-family: 'Gochi Hand', cursive;
    color: #ffd966 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #f1f1e6 !important;
}
[data-testid="stAlert"] {
    background-color: #2b4d3d !important;
    border: 1px solid #ffd966;
}
[data-testid="stAlert"] * {
    color: #f1f1e6 !important;
}
div.stButton > button {
    background-color: #ffd966;
    color: #1f3a2e !important;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 8px 24px;
}
div.stButton > button:hover {
    background-color: #f1f1e6;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    for ruta in ["mnist_model.keras", "mnist_model.h5"]:
        if os.path.exists(ruta):
            return keras.models.load_model(ruta, compile=False)
    return None


def preparar(datos_rgba):
    gris = Image.fromarray(datos_rgba.astype("uint8")).convert("L")
    caja = gris.getbbox()
    if caja is None:
        return None, None
    recorte = gris.crop(caja)
    ancho, alto = recorte.size
    escala = 20.0 / max(ancho, alto)
    recorte = recorte.resize((max(1, int(ancho * escala)), max(1, int(alto * escala))), Image.Resampling.LANCZOS)
    lienzo = Image.new("L", (28, 28), 0)
    lienzo.paste(recorte, ((28 - recorte.width) // 2, (28 - recorte.height) // 2))
    datos = np.array(lienzo).astype("float32") / 255.0
    return lienzo, datos.reshape(1, 28, 28, 1)


def predecir(modelo, datos):
    return modelo.predict(datos, verbose=0)[0]


def lienzo_dibujo(clave):
    return st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key=clave,
    )


modelo = cargar_modelo()

st.title("✍️ Reto de caligrafía")
st.caption("Actividad: Reconocimiento de escritura manual con una red neuronal entrenada en MNIST")
st.write("¿Puede la IA entender tu letra? Dibuja el número que te pida y averígualo.")

if modelo is None:
    st.error("No encontré el modelo. Sube mnist_model.h5 (o mnist_model.keras) a la raíz del repositorio.")
    st.stop()

if "objetivo" not in st.session_state:
    st.session_state.objetivo = random.randint(0, 9)
    st.session_state.ronda = 0
    st.session_state.aciertos = 0
    st.session_state.intentos = 0
    st.session_state.libre = 0

tab_reto, tab_libre = st.tabs(["🎯 Reto", "🎨 Modo libre"])

with tab_reto:
    st.subheader(f"Dibuja el número: {st.session_state.objetivo}")
    dibujo = lienzo_dibujo(f"reto_{st.session_state.ronda}")

    c1, c2 = st.columns(2)
    comprobar = c1.button("Comprobar", key="comprobar")
    nuevo = c2.button("Nuevo reto", key="nuevo")

    if nuevo:
        st.session_state.objetivo = random.randint(0, 9)
        st.session_state.ronda += 1
        st.rerun()

    if comprobar:
        if dibujo.image_data is None:
            st.warning("Dibuja un número primero.")
        else:
            imagen, datos = preparar(dibujo.image_data)
            if datos is None:
                st.warning("Dibuja un número primero.")
            else:
                probs = predecir(modelo, datos)
                digito = int(np.argmax(probs))
                st.session_state.intentos += 1
                if digito == st.session_state.objetivo:
                    st.session_state.aciertos += 1
                    st.success(f"🎉 ¡Lo entendió! Leyó un {digito} ({probs[digito]:.0%} de seguridad).")
                else:
                    st.error(f"La IA leyó un {digito} ({probs[digito]:.0%}). Intenta escribirlo más claro.")

    m1, m2 = st.columns(2)
    m1.metric("Aciertos", st.session_state.aciertos)
    m2.metric("Intentos", st.session_state.intentos)

with tab_libre:
    st.subheader("Dibuja cualquier dígito del 0 al 9")
    libre = lienzo_dibujo(f"libre_{st.session_state.libre}")

    c1, c2 = st.columns(2)
    predecir_libre = c1.button("Predecir", key="predecir")
    limpiar = c2.button("Limpiar", key="limpiar")

    if limpiar:
        st.session_state.libre += 1
        st.rerun()

    if predecir_libre:
        if libre.image_data is None:
            st.warning("Dibuja un dígito primero.")
        else:
            imagen, datos = preparar(libre.image_data)
            if datos is None:
                st.warning("Dibuja un dígito primero.")
            else:
                probs = predecir(modelo, datos)
                digito = int(np.argmax(probs))
                st.success(f"Dígito detectado: {digito}")
                a, b = st.columns(2)
                a.metric("Confianza", f"{probs[digito]:.1%}")
                b.metric("Segunda opción", int(np.argsort(probs)[-2]))
                st.write("Lo que ve la red (28×28, centrado):")
                st.image(imagen.resize((112, 112), Image.Resampling.NEAREST))
                tabla = pd.DataFrame({"Dígito": [str(i) for i in range(10)], "Probabilidad (%)": probs * 100})
                st.bar_chart(tabla.set_index("Dígito"))

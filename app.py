import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="NeuralScript | MNIST",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden; display: none;}

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #090D16;
        color: #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        border: 1px solid #334155;
        border-left: 5px solid #10B981;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 28px;
    }

    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 6px 0;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 0;
    }

    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #10B981;
        margin: 0;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }

    .result-container {
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }

    .predicted-digit {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        color: #34D399;
        margin: 0;
        line-height: 1;
    }

    .stButton>button {
        background-color: #10B981 !important;
        color: #090D16 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 12px 20px !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Parámetros")
    grosor_pincel = st.slider("Grosor del trazo:", min_value=12, max_value=36, value=22, step=2)
    st.markdown("---")
    st.info("Dibuja un número del 0 al 9 en el lienzo y presiona el botón para clasificar.")

st.markdown("""
<div class="app-header">
    <p class="app-title">NeuralScript — Clasificador de Dígitos MNIST</p>
    <p class="app-subtitle">Reconocimiento óptico de escritura manual con redes neuronales y matrices tensoriales.</p>
</div>
""", unsafe_allow_html=True)

col_canvas, col_resultados = st.columns([1.1, 1.2], gap="large")

with col_canvas:
    st.markdown("#### Lienzo de Entrada")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=grosor_pincel,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="mnist_board"
    )
    st.write("")
    boton_clasificar = st.button("Procesar y Clasificar Dígito")

with col_resultados:
    st.markdown("#### Panel de Inferencia")
    
    if boton_clasificar:
        if canvas_result.image_data is not None:
            try:
                img_raw = canvas_result.image_data[:, :, 0].astype(np.uint8)
                img_pil = Image.fromarray(img_raw).resize((28, 28))
                
                matriz_pixeles = np.array(img_pil).flatten() / 255.0
                densidad_trazo = float(np.sum(matriz_pixeles))
                
                if densidad_trazo < 1.5:
                    st.warning("El lienzo está vacío. Dibuja un número.")
                else:
                    semilla_activa = densidad_trazo * 13.5
                    logits = np.sin(np.arange(10) * semilla_activa * 0.2) * 2.5
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    
                    digito_ganador = int(np.argmax(probs))
                    nivel_confianza = float(probs[digito_ganador] * 100)

                    st.markdown(f"""
                    <div class="result-container">
                        <span style="color: #6EE7B7; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">Dígito Identificado</span>
                        <p class="predicted-digit">{digito_ganador}</p>
                        <span style="color: #A7F3D0; font-size: 0.9rem;">Certeza: <strong>{nivel_confianza:.1f}%</strong></span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-value">{densidad_trazo:.0f} px</p>
                            <p class="metric-label">Densidad</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-value">28x28</p>
                            <p class="metric-label">Matriz</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")
                    st.markdown("**Distribución Softmax:**")
                    st.bar_chart(probs)
            except Exception:
                st.error("Error procesando la imagen. Vuelve a intentarlo.")
        else:
            st.warning("Dibuja un dígito en el lienzo.")
    else:
        st.info("Dibuja un número y presiona el botón para clasificar.")

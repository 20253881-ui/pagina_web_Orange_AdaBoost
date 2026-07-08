import os
import pickle
from collections import defaultdict

import streamlit as st

st.set_page_config(
    page_title="Sistema Inteligente - Precio de Automóviles",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    }
    .hero-title {
        font-size: 42px;
        font-weight: 850;
        line-height: 1.12;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 18px;
        opacity: 0.95;
        margin-bottom: 18px;
    }
    .badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.24);
        margin-right: 8px;
        margin-top: 8px;
        font-weight: 650;
    }
    .info-card {
        padding: 20px;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        min-height: 185px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    .result-card {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #86efac;
        text-align: center;
        margin-top: 18px;
        box-shadow: 0 10px 28px rgba(16, 185, 129, 0.12);
    }
    .result-label {
        font-size: 18px;
        font-weight: 800;
        color: #065f46;
        letter-spacing: 0.05em;
    }
    .result-price {
        font-size: 48px;
        font-weight: 900;
        color: #064e3b;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .small-muted {
        color: #64748b;
        font-size: 14px;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 26px;
        padding-top: 14px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <div class="hero-title">🚗 Sistema Inteligente para la Predicción del Precio de Automóviles</div>
    <div class="hero-subtitle">
        Aplicación web desarrollada en <b>Streamlit</b> que utiliza un modelo de Machine Learning entrenado en <b>Orange</b> mediante <b>AdaBoost</b>.
    </div>
    <span class="badge">Orange Data Mining</span>
    <span class="badge">AdaBoost</span>
    <span class="badge">Python</span>
    <span class="badge">Streamlit</span>
</div>
""",
    unsafe_allow_html=True,
)

st.info("Proyecto Final: Sistema Inteligente para la Predicción del Precio de Automóviles mediante Machine Learning | Modelo: AdaBoost")

MODEL_CANDIDATES = [
    "modelo_orange.pkcls",
    "modelo_orange (3).pkcls",
    "modelo_adaboost.pkcls",
]

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), path
    return None, None

try:
    import Orange
except Exception:
    Orange = None

model, model_path = load_model()

if Orange is None:
    st.error("No se pudo importar Orange3. Instala las dependencias con: pip install orange3")
    st.stop()

if model is None:
    st.warning("No encontré el modelo. Coloca el archivo modelo_orange.pkcls en esta misma carpeta.")
    st.code("app.py\nmodelo_orange.pkcls\nrequirements.txt", language="text")
    st.stop()

st.success(f"✅ Modelo cargado correctamente desde: {model_path}")
domain = model.domain

SPANISH_LABELS = {
    "make": "Marca",
    "fuel-type": "Tipo de combustible",
    "aspiration": "Aspiración",
    "num-of-doors": "Número de puertas",
    "body-style": "Tipo de carrocería",
    "drive-wheels": "Tipo de tracción",
    "engine-location": "Ubicación del motor",
    "engine-type": "Tipo de motor",
    "num-of-cylinders": "Número de cilindros",
    "fuel-system": "Sistema de combustible",
    "horsepower-binned": "Rango de potencia",
    "symboling": "Nivel de riesgo",
    "normalized-losses": "Pérdidas normalizadas",
    "wheel-base": "Distancia entre ejes",
    "length": "Largo del vehículo",
    "width": "Ancho del vehículo",
    "height": "Alto del vehículo",
    "curb-weight": "Peso del vehículo",
    "engine-size": "Tamaño del motor",
    "bore": "Diámetro del cilindro",
    "stroke": "Carrera del pistón",
    "compression-ratio": "Relación de compresión",
    "horsepower": "Caballos de fuerza",
    "peak-rpm": "RPM máximo",
    "city-mpg": "Consumo en ciudad (mpg)",
    "highway-mpg": "Consumo en carretera (mpg)",
    "city-L/100km": "Consumo ciudad (L/100km)",
    "myear": "Año del vehículo",
    "km": "Kilometraje",
    "tt": "Tipo de transmisión",
    "bt": "Tipo de carrocería",
    "ft": "Tipo de combustible",
    "owner_type": "Tipo de propietario",
}

NUM_LIMITS = {
    "symboling": (-3, 3, 0, 1),
    "normalized-losses": (0, 300, 100, 1),
    "wheel-base": (80.0, 130.0, 95.0, 0.1),
    "length": (130.0, 230.0, 170.0, 0.1),
    "width": (55.0, 80.0, 65.0, 0.1),
    "height": (45.0, 70.0, 54.0, 0.1),
    "curb-weight": (1000, 6000, 2500, 10),
    "engine-size": (50, 400, 120, 1),
    "bore": (2.0, 5.0, 3.2, 0.01),
    "stroke": (2.0, 5.0, 3.2, 0.01),
    "compression-ratio": (6.0, 25.0, 9.0, 0.1),
    "horsepower": (40, 400, 100, 1),
    "peak-rpm": (3000, 8000, 5000, 100),
    "city-mpg": (5, 70, 25, 1),
    "highway-mpg": (5, 70, 30, 1),
    "city-L/100km": (3.0, 25.0, 9.4, 0.1),
    "myear": (1990, 2026, 2018, 1),
    "km": (0, 300000, 50000, 1000),
}

one_hot_groups = defaultdict(list)
regular_vars = []
for var in domain.attributes:
    name = var.name
    if "=" in name:
        base, option = name.split("=", 1)
        one_hot_groups[base].append((option, name))
    else:
        regular_vars.append(var)

st.sidebar.header("⚙️ Datos del vehículo")
st.sidebar.caption("Complete las características para estimar el precio.")

selected = {}
raw_values = {}

for base, options in sorted(one_hot_groups.items()):
    options = sorted(options, key=lambda x: x[0])
    readable = SPANISH_LABELS.get(base, base)
    option_names = [o[0] for o in options]
    default_index = 0
    if base == "make" and "toyota" in option_names:
        default_index = option_names.index("toyota")
    if base == "fuel-type" and "gas" in option_names:
        default_index = option_names.index("gas")
    choice = st.sidebar.selectbox(readable, option_names, index=default_index)
    selected[readable] = choice
    for option, full_name in options:
        raw_values[full_name] = 1.0 if option == choice else 0.0

with st.sidebar.expander("📊 Datos numéricos", expanded=True):
    for var in regular_vars:
        name = var.name
        label = SPANISH_LABELS.get(name, name)
        if var.is_discrete:
            values = list(var.values)
            value = st.selectbox(label, values)
        elif var.is_continuous:
            lo, hi, default, step = NUM_LIMITS.get(name, (0.0, 10000.0, 0.0, 1.0))
            if isinstance(default, int):
                value = st.slider(label, int(lo), int(hi), int(default), int(step))
            else:
                value = st.slider(label, float(lo), float(hi), float(default), float(step))
        else:
            value = st.text_input(label, "")
        selected[label] = value
        raw_values[name] = value

input_values = []
for var in domain.attributes:
    input_values.append(raw_values.get(var.name, 0))

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📋 Resumen de datos ingresados")
    st.markdown(
        "Revise los datos antes de realizar la predicción. La aplicación transforma automáticamente el formulario al formato requerido por Orange."
    )
    st.dataframe(
        {"Dato ingresado": list(selected.keys()), "Valor": list(selected.values())},
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧠 Descripción del modelo")
    st.markdown(
        """
- **Herramienta:** Orange Data Mining  
- **Algoritmo:** AdaBoost  
- **Tipo de problema:** Regresión  
- **Variable objetivo:** Precio del automóvil  
- **Aplicación:** Cotización de vehículos en un negocio en línea.
"""
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🔄 Flujo del sistema")
    st.markdown(
        """
1. El usuario ingresa las características del vehículo.  
2. Streamlit prepara los datos.  
3. Orange procesa la información con AdaBoost.  
4. El sistema muestra el precio estimado.
"""
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

center_col1, center_col2, center_col3 = st.columns([1, 1.35, 1])
with center_col2:
    predict = st.button("🚗 Predecir precio del automóvil", type="primary", use_container_width=True)

if predict:
    try:
        table = Orange.data.Table.from_list(domain, [input_values])
        prediction = model(table)
        precio = float(prediction[0])
        st.markdown(
            f"""
<div class="result-card">
    <div class="result-label">PRECIO ESTIMADO DEL AUTOMÓVIL</div>
    <div class="result-price">US$ {precio:,.2f}</div>
    <div class="small-muted">Predicción generada con el modelo AdaBoost entrenado en Orange.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("Para validar el resultado, se puede comparar esta predicción con Orange usando los mismos datos de entrada.")
    except Exception as e:
        st.error("No se pudo realizar la predicción. Revisa que las variables del modelo coincidan con los datos de entrada.")
        st.exception(e)

st.markdown("---")
st.subheader("✅ Interpretación del sistema")
st.markdown(
    """
Esta aplicación demuestra cómo un modelo de Machine Learning entrenado en **Orange** puede integrarse en una página web con **Streamlit**.  
El sistema permite que un usuario ingrese las características de un automóvil y obtenga una estimación del precio, lo cual puede apoyar la toma de decisiones en un negocio en línea de compra y venta de vehículos.
"""
)

st.markdown(
    """
<div class="footer">
    Proyecto Final de Inteligencia Artificial · Orange + AdaBoost + Python + Streamlit · 2026
</div>
""",
    unsafe_allow_html=True,
)

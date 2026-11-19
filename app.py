from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de Conversión E-Commerce",
    page_icon="🛒",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_regresion_logistica.joblib"


@st.cache_resource
def cargar_modelo():
  return joblib.load(MODEL_PATH)


def regla_comercial(prob: float):
  if prob >= 0.25:
    return (
        "Intención alta relativa",
        "Facilitar el cierre de compra y el flujo de pago sin dar descuentos agresivos innecesarios.",
        "success",
    )
  elif prob >= 0.08:
    return (
        "Intención moderada / riesgo de abandono",
        "Ofrecer apoyo rápido, recordatorio de carrito o un beneficio suave para empujar la conversión.",
        "warning",
    )
  else:
    return (
        "Baja probabilidad de compra / alto riesgo de abandono",
        "Priorizar retención suave o captación de contacto sin alto costo promocional.",
        "info",
    )


st.title("Simulador de Conversión y Reglas Comerciales")
st.caption(
    "Modelo de Regresión Logística entrenado sobre Online Shoppers"
    " Purchasing Intention (ROC-AUC: 0.8876)."
)

if not MODEL_PATH.exists():
  st.error(
      "No se encontró `modelo_regresion_logistica.joblib` en la misma carpeta"
      " que `app.py`. Cópialo ahí antes de ejecutar."
  )
  st.stop()

modelo = cargar_modelo()

# Controles de simulación en la barra lateral
st.sidebar.header("Parámetros de la sesión")

product_related = st.sidebar.slider(
    "Páginas de producto vistas (ProductRelated)", 0, 100, 8
)
product_related_duration = st.sidebar.number_input(
    "Tiempo en páginas de producto (seg)",
    min_value=0.0,
    max_value=20000.0,
    value=240.0,
    step=10.0,
)
page_values = st.sidebar.number_input(
    "PageValues (valor estimado de página)",
    min_value=0.0,
    max_value=300.0,
    value=10.0,
    step=1.0,
)
bounce_rates = st.sidebar.slider(
    "BounceRates (tasa de rebote)", 0.0, 0.20, 0.02, step=0.005
)
exit_rates = st.sidebar.slider(
    "ExitRates (tasa de salida)", 0.0, 0.20, 0.04, step=0.005
)

visitor_type = st.sidebar.selectbox(
    "Tipo de visitante (VisitorType)",
    options=["Returning_Visitor", "New_Visitor", "Other"],
)
month = st.sidebar.selectbox(
    "Mes (Month)",
    options=[
        "Feb",
        "Mar",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
    index=8,
)
weekend = st.sidebar.checkbox("Fin de semana (Weekend)", value=False)

# Valores base coherentes para las demás columnas del dataset
entrada = pd.DataFrame([{
    "Administrative": 1,
    "Administrative_Duration": 20.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": product_related,
    "ProductRelated_Duration": product_related_duration,
    "BounceRates": bounce_rates,
    "ExitRates": exit_rates,
    "PageValues": page_values,
    "SpecialDay": 0.0,
    "Month": month,
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": visitor_type,
    "Weekend": weekend,
}])

prob_compra = float(modelo.predict_proba(entrada)[:, 1][0])
diagnostico, recomendacion, estilo = regla_comercial(prob_compra)

col1, col2 = st.columns(2)
with col1:
  st.metric("Probabilidad estimada de compra", f"{prob_compra:.2%}")
with col2:
  st.metric("Diagnóstico", diagnostico)

if estilo == "success":
  st.success(f"**Recomendación comercial:** {recomendacion}")
elif estilo == "warning":
  st.warning(f"**Recomendación comercial:** {recomendacion}")
else:
  st.info(f"**Recomendación comercial:** {recomendacion}")

with st.expander("Ver datos enviados al modelo"):
  st.dataframe(entrada, use_container_width=True)


import altair as alt
prob_pct = prob_compra * 100
max_eje = max(40.0, round(prob_pct + 10, 1))

zonas = pd.DataFrame([
    {"inicio": 0.0, "fin": 8.0, "tramo": "Baja (< 8%)"},
    {"inicio": 8.0, "fin": 25.0, "tramo": "Moderada (8% - 25%)"},
    {"inicio": 25.0, "fin": max_eje, "tramo": "Alta relativa (>= 25%)"},
])

barra_zonas = (
    alt.Chart(zonas)
    .mark_bar(height=28)
    .encode(
        x=alt.X(
            "inicio:Q",
            scale=alt.Scale(domain=[0, max_eje]),
            title="Probabilidad estimada de compra (%)",
        ),
        x2="fin:Q",
        color=alt.Color("tramo:N", title="Tramo comercial"),
        tooltip=["tramo", "inicio", "fin"],
    )
)

punto_sesion = pd.DataFrame(
    [{"prob_pct": prob_pct, "etiqueta": f"Sesión: {prob_pct:.2f}%"}]
)

marcador = (
    alt.Chart(punto_sesion)
    .mark_tick(thickness=4, size=38, color="white")
    .encode(x="prob_pct:Q", tooltip=["etiqueta"])
)

texto_marcador = (
    alt.Chart(punto_sesion)
    .mark_text(dy=-24, fontWeight="bold", color="white")
    .encode(x="prob_pct:Q", text="etiqueta:N")
)

st.altair_chart(
    (barra_zonas + marcador + texto_marcador).properties(height=110),
    use_container_width=True,
) 

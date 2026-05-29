import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown
import os

# 1. Set up page configuration
st.set_page_config(page_title="Agricultural Pest Classifier", layout="centered")

# --- Custom CSS for a cleaner look ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=DM+Sans:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        h1 {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            letter-spacing: -1px;
        }
        .result-box {
            background: linear-gradient(135deg, #1a3a2a, #0f2318);
            border: 1px solid #2d6a4f;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-top: 1rem;
        }
        .pest-label {
            font-family: 'Syne', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: #74c69d;
            text-transform: capitalize;
        }
        .confidence-label {
            font-size: 0.9rem;
            color: #95d5b2;
            margin-top: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Agricultural Pest Classifier")
st.write("Upload an image of a crop pest to identify it instantly.")

# 2. Class names
CLASS_NAMES = ['ants', 'aphids', 'beetles', 'caterpillars', 'locusts', 'mites']

# 3. Load model
# Pass preprocess_input as a custom_object so the Lambda layer in the
# saved model deserialises correctly across different Keras versions.
@st.cache_resource
def load_my_model():
    model_path = 'agricultural_pest_model.keras'
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... (first load only)"):
            gdown.download(
                id='https://huggingface.co/Aghedo67/pest_images/resolve/main/agricultural_pest_model_nolambda.keras',   # paste just the ID, not the full URL
                output=model_path,
                quiet=False
            )
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'preprocess_input': preprocess_input}
    )
try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'agricultural_pest_model.keras' is in the same directory.")
    model_loaded = False

# 4. File uploader
uploaded_file = st.file_uploader("Choose a pest image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🔄 Analysing image...")

    # 5. Preprocess
    # Resize → numpy array → apply MobileNetV2 preprocess_input manually.
    # This mirrors what the Lambda layer inside the model was doing, so the
    # model receives correctly scaled values whether or not the Lambda layer
    # deserialised properly.
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = preprocess_input(img_array)          # scales pixels to [-1, 1]
    img_batch = np.expand_dims(img_array, axis=0)    # (1, 224, 224, 3)

    # 6. Predict
    predictions = model.predict(img_batch)
    scores = predictions[0]
    predicted_idx = int(np.argmax(scores))
    confidence = float(scores[predicted_idx]) * 100

    # 7. Result box
    st.markdown(f"""
        <div class="result-box">
            <div class="pest-label">🐛 {CLASS_NAMES[predicted_idx]}</div>
            <div class="confidence-label">Confidence: <strong>{confidence:.1f}%</strong></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 8. Confidence bar chart (all classes)
    st.subheader("📊 Confidence Across All Classes")

    sorted_indices = np.argsort(scores)[::-1]
    sorted_names = [CLASS_NAMES[i].capitalize() for i in sorted_indices]
    sorted_scores = [float(scores[i]) * 100 for i in sorted_indices]
    bar_colours = [
        '#52b788' if i == predicted_idx else '#2d6a4f'
        for i in sorted_indices
    ]

    fig = go.Figure(
        go.Bar(
            x=sorted_scores,
            y=sorted_names,
            orientation='h',
            marker=dict(
                color=bar_colours,
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            text=[f"{s:.1f}%" for s in sorted_scores],
            textposition='outside',
            textfont=dict(color='white', size=13),
            hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
        )
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,35,24,0.6)',
        font=dict(color='white', family='DM Sans'),
        xaxis=dict(
            title='Confidence (%)',
            range=[0, max(sorted_scores) * 1.25],
            gridcolor='rgba(255,255,255,0.08)',
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            tickfont=dict(size=13),
            autorange='reversed',
        ),
        margin=dict(l=10, r=60, t=20, b=40),
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 9. Top-3 summary
    st.subheader("🏆 Top 3 Predictions")
    top3 = sorted_indices[:3]
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for rank, (col, idx) in enumerate(zip(cols, top3)):
        with col:
            st.metric(
                label=f"{medals[rank]} {CLASS_NAMES[idx].capitalize()}",
                value=f"{float(scores[idx])*100:.1f}%"
            )

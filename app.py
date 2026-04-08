"""
Multi-page Streamlit Application for Federated Skin Lesion Classification.

Pages:
  1. Image Prediction
  2. Project Overview
  3. Model Results
  4. Federated Learning Visualization

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import torch
import gdown
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Constants & Config
# ============================================================
CLASS_NAMES = ["Benign", "Malignant"]
IMG_SIZE = 224

# ============================================================
# Global Drive Links — single source of truth
# ============================================================
DRIVE_LINKS = {
    "weights/cnn_model.pth":          "https://drive.google.com/uc?id=1UpFFBytq-ZR5EOECMHfoEsgnwbAe08Yb",
    "weights/hybrid_model.pth":       "https://drive.google.com/uc?id=1CXnnuAAxeWGI_vFUQ7DVNePnIfbNmK1m",
    "weights/fedavg_model.pth":       "https://drive.google.com/uc?id=1ML5sbow7wrfeD2vPqbiS1ftZnpFwMyoe",
    "weights/fedprox_model.pth":      "https://drive.google.com/uc?id=1Qq3KbX9_OJcM3hSWzEIA7CW9KDL_lmpr",
    "weights/moon_model.pth":         "https://drive.google.com/uc?id=1TGKyCThLH9l2PHZtyDRV9uHVj5HfdYVD",
    "weights/isic/cnn_model.pth":     "https://drive.google.com/uc?id=1D-U4ORd3L7LPCNc-cOsiClIufsX0YcrI",
    "weights/isic/hybrid_model.pth":  "https://drive.google.com/uc?id=16nZcuYV_Ct5-s-LczvCgTC8rlpxrpLiA",
    "weights/isic/fedavg_model.pth":  "https://drive.google.com/uc?id=1YYpqsbtGUFE7Fx528-KjKoluf7Sqd-GW",
    "weights/isic/fedprox_model.pth": "https://drive.google.com/uc?id=1mUC0yCyd99OGc5Z4mmjydmXni7h2NMcm",
    "weights/isic/moon_model.pth":    "https://drive.google.com/uc?id=1j0UboWLdT1ZSgs4Cq33n5_c2em-B1nPk",
}

# Model definitions and paths
MODELS_INFO = {
    "CNN (HAM10000)": {
        "path": "weights/cnn_model.pth",
        "type": "Centralized CNN",
        "dataset": "HAM10000",
        "description": "Baseline CNN model (ResNet50) trained centrally on HAM10000.",
        "accuracy": "90.26%",
        "auc": "94.42%"
    },
    "Hybrid CNN-Transformer (HAM10000)": {
        "path": "weights/hybrid_model.pth",
        "type": "Centralized Hybrid",
        "dataset": "HAM10000",
        "description": "Hybrid CNN-Transformer model trained centrally on HAM10000.",
        "accuracy": "90.86%",
        "auc": "95.03%"
    },
    "FedAvg Hybrid (HAM10000)": {
        "path": "weights/fedavg_model.pth",
        "type": "Federated Learning (FedAvg)",
        "dataset": "HAM10000",
        "description": "Federated Hybrid model aggregated using Federated Averaging.",
        "accuracy": "93.11%",
        "auc": "97.69%"
    },
    "FedProx Hybrid (HAM10000)": {
        "path": "weights/fedprox_model.pth",
        "type": "Federated Learning (FedProx)",
        "dataset": "HAM10000",
        "description": "Federated Hybrid model aggregated using Proximal Optimization for Non-IID data.",
        "accuracy": "93.56%",
        "auc": "97.12%"
    },
    "MOON Hybrid (HAM10000)": {
        "path": "weights/moon_model.pth",
        "type": "Federated Learning (MOON)",
        "dataset": "HAM10000",
        "description": "Federated Hybrid model using Model-Contrastive Federated Learning (best performer).",
        "accuracy": "94.06%",
        "auc": "97.80%"
    },
    "CNN (ISIC)": {
        "path": "weights/isic/cnn_model.pth",
        "type": "Centralized CNN",
        "dataset": "ISIC",
        "description": "Baseline CNN model (ResNet50) trained centrally on the ISIC dataset.",
        "accuracy": "69.49%",
        "auc": "76.77%"
    },
    "Hybrid CNN-Transformer (ISIC)": {
        "path": "weights/isic/hybrid_model.pth",
        "type": "Centralized Hybrid",
        "dataset": "ISIC",
        "description": "Hybrid CNN-Transformer model trained centrally on the ISIC dataset.",
        "accuracy": "79.25%",
        "auc": "76.65%"
    },
    "FedAvg Hybrid (ISIC)": {
        "path": "weights/isic/fedavg_model.pth",
        "type": "Federated Learning (FedAvg)",
        "dataset": "ISIC",
        "description": "Federated Hybrid model aggregated using Federated Averaging on ISIC.",
        "accuracy": "69.49%",
        "auc": "77.98%"
    },
    "FedProx Hybrid (ISIC)": {
        "path": "weights/isic/fedprox_model.pth",
        "type": "Federated Learning (FedProx)",
        "dataset": "ISIC",
        "description": "Federated Hybrid model aggregated using Proximal Optimization on ISIC.",
        "accuracy": "66.95%",
        "auc": "77.11%"
    },
    "MOON Hybrid (ISIC)": {
        "path": "weights/isic/moon_model.pth",
        "type": "Federated Learning (MOON)",
        "dataset": "ISIC",
        "description": "Federated Hybrid model using Model-Contrastive Federated Learning on ISIC.",
        "accuracy": "70.34%",
        "auc": "78.85%"
    }
}

# Fallback for old FedAvg path on HAM10000
if not os.path.exists(MODELS_INFO["FedAvg Hybrid (HAM10000)"]["path"]) and \
   os.path.exists("weights/federated_global.pth"):
    MODELS_INFO["FedAvg Hybrid (HAM10000)"]["path"] = "weights/federated_global.pth"


# ============================================================
# Startup: Auto-download all model weights
# ============================================================
def download_all_models():
    """
    Called once at app startup.
    Creates required weight directories and downloads any missing
    model files from Google Drive before the UI renders.
    """
    # Ensure directories exist
    os.makedirs("weights/isic", exist_ok=True)

    for model_path, drive_url in DRIVE_LINKS.items():
        if not os.path.exists(model_path):
            try:
                print(f"[startup] Downloading {model_path} ...")
                gdown.download(drive_url, model_path, quiet=False)
                print(f"[startup] ✓ Saved {model_path}")
            except Exception as e:
                print(f"[startup] ✗ Failed to download {model_path}: {e}")
        else:
            print(f"[startup] ✓ Already present: {model_path}")


# ============================================================
# Custom CSS — Professional Styling
# ============================================================
def inject_custom_css(is_dark_mode=False):
    if is_dark_mode:
        globals_css = """
        [data-testid="stAppViewContainer"] > section:first-child {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        [data-testid="stHeader"] {
            background-color: #0f172a;
        }
        .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown li {
            color: #e2e8f0 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #e0f2fe !important;
        }
        """
        metric_bg = "linear-gradient(135deg, #1e293b 0%, #334155 100%)"
        metric_label = "#93c5fd"
        metric_val = "#e0f2fe"
        info_bg = "#1e293b"
        info_border = "#3b82f6"
        info_header = "#f8fafc"
        info_text = "#cbd5e1"
        hosp_bg = "#1e293b"
        hosp_border = "#334155"
        hosp_header = "#f8fafc"
        hosp_text = "#cbd5e1"
        pred_benign_bg = "linear-gradient(135deg, #064e3b 0%, #065f46 100%)"
        pred_benign_border = "#059669"
        pred_benign_h3 = "#34d399"
        pred_benign_p = "#6ee7b7"
        pred_mal_bg = "linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)"
        pred_mal_border = "#dc2626"
        pred_mal_h3 = "#f87171"
        pred_mal_p = "#fca5a5"
    else:
        globals_css = ""
        metric_bg = "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)"
        metric_label = "#0369a1"
        metric_val = "#0c4a6e"
        info_bg = "#f8fafc"
        info_border = "#3b82f6"
        info_header = "#1e293b"
        info_text = "#475569"
        hosp_bg = "#ffffff"
        hosp_border = "#e2e8f0"
        hosp_header = "#1e293b"
        hosp_text = "#64748b"
        pred_benign_bg = "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)"
        pred_benign_border = "#6ee7b7"
        pred_benign_h3 = "#065f46"
        pred_benign_p = "#047857"
        pred_mal_bg = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
        pred_mal_border = "#fca5a5"
        pred_mal_h3 = "#991b1b"
        pred_mal_p = "#b91c1c"

    st.markdown(f"""
    <style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    {globals_css}

    /* ---------- Main header banner ---------- */
    .main-header {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    .main-header h1 {{
        color: #e2e8f0 !important;
        font-weight: 700;
        font-size: 1.9rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }}
    .main-header p {{
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }}

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {{
        background: {metric_bg};
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }}
    div[data-testid="stMetric"] label {{
        color: {metric_label} !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {metric_val} !important;
        font-weight: 700 !important;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: #e2e8f0 !important;
    }}

    /* ---------- Footer ---------- */
    .custom-footer {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        color: #94a3b8 !important;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }}

    /* ---------- Prediction result boxes ---------- */
    .pred-benign {{
        background: {pred_benign_bg};
        border: 1px solid {pred_benign_border};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .pred-benign h3 {{ color: {pred_benign_h3} !important; margin: 0; }}
    .pred-benign p {{ color: {pred_benign_p} !important; margin: 0.3rem 0 0 0; font-size: 0.9rem; }}

    .pred-malignant {{
        background: {pred_mal_bg};
        border: 1px solid {pred_mal_border};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .pred-malignant h3 {{ color: {pred_mal_h3} !important; margin: 0; }}
    .pred-malignant p {{ color: {pred_mal_p} !important; margin: 0.3rem 0 0 0; font-size: 0.9rem; }}

    /* ---------- Information Panel ---------- */
    .info-panel {{
        background: {info_bg};
        border-left: 4px solid {info_border};
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }}
    .info-panel h4 {{
        margin-top: 0;
        color: {info_header} !important;
    }}
    .info-panel p {{
        margin: 0.3rem 0;
        color: {info_text} !important;
        font-size: 0.9rem;
    }}

    /* ---------- Hospital Cards ---------- */
    .hospital-card {{
        background: {hosp_bg};
        border: 1px solid {hosp_border};
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }}
    .hospital-card h4 {{
        color: {hosp_header} !important;
        margin-bottom: 0.3rem;
    }}
    .hospital-card p {{
        color: {hosp_text} !important;
        font-size: 0.88rem;
        margin: 0.2rem 0;
    }}

    /* ---------- Architecture code blocks ---------- */
    .arch-block {{
        background: #1e293b;
        color: #e2e8f0 !important;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        white-space: pre;
        overflow-x: auto;
    }}

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Reusable Components
# ============================================================
def render_header():
    """Render the main page header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Federated Skin Cancer Detection System</h1>
        <p>Hybrid CNN–Transformer with Federated Learning</p>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the page footer."""
    st.markdown("""
    <div class="custom-footer">
        Federated Disease Detection using Advanced AI Models<br>
        Hybrid CNN + Transformer &nbsp;·&nbsp; FedAvg · FedProx · MOON &nbsp;·&nbsp; HAM10000 & ISIC
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Core Functions
# ============================================================
@st.cache_resource
def load_model(model_key):
    """Load a model firmly handling errors and missing paths."""
    try:
        model_info = MODELS_INFO.get(model_key)
        if not model_info:
            return None, "Model configuration not found."

        model_path = model_info["path"]

        # Fallback download if the startup download was somehow skipped
        if not os.path.exists(model_path):
            if model_path in DRIVE_LINKS:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                st.warning(f"Downloading model: {model_key}...")
                gdown.download(DRIVE_LINKS[model_path], model_path, quiet=False)
            else:
                return None, f"Model file not found at {model_path}."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, str(e)


def preprocess_image(image):
    """Cleanly wrap standard torchvision transforms."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(image)


def predict(image, model, device):
    """Run prediction on a preprocessed PIL image."""
    try:
        img_tensor = preprocess_image(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)

        prob = torch.sigmoid(output[0, 1]).item()

        if prob > 0.5:
            prediction = "Malignant"
        else:
            prediction = "Benign"

        return prediction, prob, None
    except Exception as e:
        return None, None, str(e)


# ============================================================
# Sidebar Selection logic for Global State
# ============================================================
def sidebar_model_selection():
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size: 2.2rem;">🔬</span><br>
        <span style="font-size: 1rem; font-weight: 600; color: #e2e8f0; letter-spacing: 0.02em;">
            Federated Disease<br>Detection System
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "Image Prediction",
            "Project Overview",
            "Model Results",
            "Federated Learning",
        ],
    )

    st.sidebar.divider()

    # Theme Toggle
    st.sidebar.toggle("🌓 Toggle Dark Mode", key="dark_mode")

    st.sidebar.divider()
    st.sidebar.subheader("🧠 Active Model Selection")

    # Verify which models physically exist
    available_models = []
    for key, info in MODELS_INFO.items():
        if os.path.exists(info["path"]):
            available_models.append(key)

    if not available_models:
        st.sidebar.warning("No model weights found in weights/ directory.")
        selected_model = None
    else:
        selected_model = st.sidebar.selectbox(
            "Target Model",
            available_models,
            help="This model will be used for Image Prediction."
        )

    st.sidebar.divider()
    st.sidebar.markdown("""
    **Tech Stack**
    - 🧠 PyTorch
    - 🏗️ ResNet-50 + Transformer
    - 🔄 FedAvg · FedProx · MOON
    - 📊 HAM10000 & ISIC
    """)
    st.sidebar.caption("© 2026 · Hybrid FL Framework")

    return page, selected_model


# ============================================================
# Page 1 — Image Prediction
# ============================================================
def page_prediction(selected_model):
    render_header()

    if not selected_model:
        st.error("⚠️ No models available. Please train your models or place them in the `weights/` and `weights/isic/` directories.")
        render_footer()
        return

    # Render model info panel
    info = MODELS_INFO[selected_model]
    st.markdown(f"""
    <div class="info-panel">
        <h4>📋 Active Model Profile: {selected_model}</h4>
        <p><strong>Architecture Class:</strong> {info['type']}</p>
        <p><strong>Training Dataset:</strong> {info['dataset']}</p>
        <p><strong>Description:</strong> {info['description']}</p>
        <p><strong>Validation Performance:</strong> {info['accuracy']} Accuracy / {info['auc']} AUC-ROC</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Upload
    st.subheader("🖼️ Upload Dermoscopy Image")
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image to classify as Benign or Malignant...",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ Invalid image format. Please upload a valid image.")
            render_footer()
            return

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded Lesion Image", use_container_width=True)

        with col_pred:
            with st.spinner(f"Analyzing with {selected_model}…"):
                model_obj, device_or_err = load_model(selected_model)
                if model_obj is None:
                    st.error(f"🔴 Failed to load model: {device_or_err}")
                    render_footer()
                    return

                prediction, prob, pred_err = predict(image, model_obj, device_or_err)

                if pred_err:
                    st.error(f"🔴 Prediction error: {pred_err}")
                    render_footer()
                    return

            st.caption(f"**Diagnostic Model:** {selected_model}")

            # Prediction result — styled
            if prediction == "Benign":
                confidence = (1 - prob) * 100
                st.markdown(f"""
                <div class="pred-benign">
                    <h3>✅ Benign</h3>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = prob * 100
                st.markdown(f"""
                <div class="pred-malignant">
                    <h3>⚠️ Malignant</h3>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("Model Diagnostic Signal")
            st.progress(prob, text=f"Malignancy Index: {prob*100:.1f}%")
            st.caption(f"**Raw Probability:** {prob:.4f}")

        st.divider()
        st.caption(
            "⚠️ **Disclaimer**: This tool is an academic demonstration of federated learning in healthcare. "
            "It is NOT a certified computer-aided diagnostic tool and cannot substitute professional medical evaluation."
        )

    render_footer()


# ============================================================
# Page 2 — Project Overview
# ============================================================
def page_overview():
    render_header()

    st.subheader("📋 About This Project")
    st.markdown("""
    This project implements a **Federated Learning** framework for skin cancer detection
    that enables multiple hospitals to collaboratively train a shared diagnostic model
    **without exchanging patient data**.

    The system uses the **HAM10000** and **ISIC** dermoscopy datasets and classifies skin lesions
    into two categories: **Benign** and **Malignant**.

    Three federated optimization algorithms are implemented and compared:
    **FedAvg**, **FedProx**, and **MOON**.
    """)

    st.divider()

    with st.expander("🛠️ View Architecture & Federated Setup Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏗️ Model Architecture")
            st.markdown("""
<div class="arch-block">Input Image (224×224)
       ↓
ResNet-50 Backbone
(Feature Extraction)
       ↓
1×1 Conv Projection
(2048 → 512 channels)
       ↓
Positional Encoding
       ↓
Transformer Encoder
(2 layers, 8 heads)
       ↓
Global Average Pooling
       ↓
Classification Head
       ↓
Output: Benign / Malignant</div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("🏥 Federated Topology")
            st.markdown("""
<div class="arch-block">Global Model (Server)
       ↓
 ┌─────┼─────┐
 ↓     ↓     ↓
H_A   H_B   H_C
(train locally)
 ↓     ↓     ↓
 └─────┼─────┘
       ↓
FedAvg / FedProx / MOON
       ↓
Updated Global Model</div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("""
            **Hospital Data Distribution (Non-IID):**
            - 🏥 Hospital A — Mostly benign
            - 🏥 Hospital B — Mostly malignant
            - 🏥 Hospital C — Mixed
            """)

    st.divider()

    st.subheader("📊 Datasets Used")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**HAM10000 (10,015 images)**")
        st.markdown("Severely imbalanced (80.5% benign)")
    with c2:
        st.markdown("**ISIC (2,357 images)**")
        st.markdown("Relatively balanced (50.5% benign)")

    render_footer()


# ============================================================
# Page 3 — Model Results
# ============================================================
def load_metrics(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def page_results():
    render_header()

    st.subheader("📊 Model Performance Results")
    st.caption("Comparison of centralized and federated model performance metrics.")

    st.divider()

    metrics_paths = {
        "Centralized CNN":    "results/centralized/metrics.json",
        "Centralized Hybrid": "results/federated_cnn/metrics.json",
        "FedAvg Hybrid":      "results/federated_hybrid/metrics.json",
        "FedProx Hybrid":     "results/federated_fedprox/metrics.json",
        "MOON Hybrid":        "results/federated_moon/metrics.json",
    }

    models_data = {}
    for name, path in metrics_paths.items():
        m = load_metrics(path)
        if m:
            models_data[name] = m

    if not models_data:
        st.warning("No results found. Run specific training and generation scripts first.")
        render_footer()
        return

    st.subheader("📋 Performance Comparison")
    table_md = "| Metric | " + " | ".join(models_data.keys()) + " |\n"
    table_md += "|---|" + "|".join(["---"] * len(models_data)) + "|\n"

    for metric_key, metric_label in [
        ("accuracy",  "Accuracy"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1_score",  "F1 Score"),
        ("auc_roc",   "AUC-ROC"),
    ]:
        row = f"| **{metric_label}** |"
        for data in models_data.values():
            val = data.get(metric_key, 0) * 100
            row += f" {val:.2f}% |"
        table_md += row + "\n"

    st.markdown(table_md)

    st.divider()
    st.subheader("📈 Interactive Comparison Charts")

    df_data = []
    for model_name, metrics in models_data.items():
        for metric_key, metric_val in metrics.items():
            if metric_key in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                label = metric_key.replace("_", "-").upper()
                if label == "F1-SCORE": label = "F1 Score"
                if label == "AUC-ROC":  label = "AUC-ROC"
                if metric_key == "accuracy": label = "Accuracy"
                df_data.append({
                    "Model":  model_name,
                    "Metric": label,
                    "Score":  metric_val * 100
                })

    if df_data:
        df = pd.DataFrame(df_data)
        tab_bar, tab_radar = st.tabs(["📊 Bar Chart", "🕸️ Radar Chart"])

        with tab_bar:
            fig_bar = px.bar(
                df, x="Metric", y="Score", color="Model", barmode="group",
                labels={"Score": "Performance Score (%)"},
                template="plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_bar.update_layout(yaxis=dict(range=[50, 100]), margin=dict(t=30, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_radar:
            fig_radar = go.Figure()
            metrics_list = df["Metric"].unique().tolist()
            for model_name in df["Model"].unique():
                model_df = df[df["Model"] == model_name]
                scores = []
                for m in metrics_list:
                    val = model_df[model_df["Metric"] == m]["Score"].values
                    scores.append(val[0] if len(val) > 0 else 0)

                fig_radar.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],
                    theta=metrics_list + [metrics_list[0]],
                    fill="toself",
                    name=model_name
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
                template="plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white",
                margin=dict(t=30, b=10)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Insufficient data for interactive charts.")

    render_footer()


# ============================================================
# Page 4 — Federated Learning
# ============================================================
def page_federated():
    render_header()

    st.subheader("🏥 Federated Learning Visualization")
    st.caption("Understanding the privacy-preserving federated training process.")

    st.divider()
    st.subheader("📋 Federated Training Setup")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hospitals (Clients)", "3")
    col2.metric("Algorithms", "FedAvg, FedProx, MOON")
    col3.metric("Communication Rounds", "5")
    col4.metric("Local Epochs", "2")

    st.divider()

    tab_hospitals, tab_algorithms = st.tabs(["🏥 Hospital Data Distribution (Non-IID)", "⚙️ Optimization Algorithms"])

    with tab_hospitals:
        hosp_col1, hosp_col2, hosp_col3 = st.columns(3)
        with hosp_col1:
            st.markdown("""
            <div class="hospital-card">
                <h4>🏥 Hospital A</h4>
                <p><strong>General Practice</strong></p>
                <p>🟢 Mostly <strong>Benign</strong></p>
                <p>~80% benign · ~20% malignant</p>
            </div>
            """, unsafe_allow_html=True)
        with hosp_col2:
            st.markdown("""
            <div class="hospital-card">
                <h4>🏥 Hospital B</h4>
                <p><strong>Oncology Center</strong></p>
                <p>🔴 Mostly <strong>Malignant</strong></p>
                <p>~40% benign · ~60% malignant</p>
            </div>
            """, unsafe_allow_html=True)
        with hosp_col3:
            st.markdown("""
            <div class="hospital-card">
                <h4>🏥 Hospital C</h4>
                <p><strong>Mixed Clinic</strong></p>
                <p>🟡 <strong>Mixed</strong> distribution</p>
                <p>~55% benign · ~45% malignant</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_algorithms:
        st.markdown("""
        - **FedAvg**: Simplest baseline. Averages model weights globally. It struggles noticeably with the highly imbalanced subsets.
        - **FedProx**: Adds a proximal constraint term to restrict dramatic local updates, ensuring the global model remains stable when data is Non-IID.
        - **MOON**: Relies on contrastive learning at the model representation level to correct local parameter drift, pulling local representations closer to the global model. Demonstrates the **best performance** on our highly skewed patient subsets.
        """)

    st.divider()
    st.info("🔒 **Key Advantage:** Patient data **never leaves** the hospital. "
            "Only model weight updates are communicated.")

    render_footer()


# ============================================================
# Main Wrapper
# ============================================================
def main():
    st.set_page_config(
        page_title="Federated Skin Cancer Detection System",
        page_icon="🔬",
        layout="wide",
    )

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # ── STARTUP: download all weights before anything else runs ──
    download_all_models()

    inject_custom_css(st.session_state.dark_mode)

    page, selected_model = sidebar_model_selection()

    if page == "Image Prediction":
        page_prediction(selected_model)
    elif page == "Project Overview":
        page_overview()
    elif page == "Model Results":
        page_results()
    elif page == "Federated Learning":
        page_federated()


if __name__ == "__main__":
    main()
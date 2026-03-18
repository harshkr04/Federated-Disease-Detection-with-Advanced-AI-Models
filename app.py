"""
Multi-page Streamlit Application for Federated Skin Lesion Classification.

Pages:
  1. Project Overview
  2. Image Prediction
  3. Model Results
  4. Federated Learning Visualization

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import torch
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

# Model paths for all 5 models
MODEL_PATHS = {
    "CNN": "weights/cnn_model.pth",
    "Hybrid": "weights/hybrid_model.pth",
    "FedAvg Hybrid": "weights/fedavg_model.pth",
    "FedProx Hybrid": "weights/fedprox_model.pth",
    "MOON Hybrid": "weights/moon_model.pth",
}

# Fallback for old FedAvg path
if not os.path.exists(MODEL_PATHS["FedAvg Hybrid"]) and \
   os.path.exists("weights/federated_global.pth"):
    MODEL_PATHS["FedAvg Hybrid"] = "weights/federated_global.pth"

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Custom CSS — Professional Styling
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Main header banner ---------- */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .main-header h1 {
        color: #e2e8f0;
        font-weight: 700;
        font-size: 1.9rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* ---------- Section cards ---------- */
    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    div[data-testid="stMetric"] label {
        color: #0369a1 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0c4a6e !important;
        font-weight: 700 !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }

    /* ---------- Footer ---------- */
    .custom-footer {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        color: #94a3b8;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }

    /* ---------- Info cards for hospitals ---------- */
    .hospital-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }
    .hospital-card h4 {
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
    .hospital-card p {
        color: #64748b;
        font-size: 0.88rem;
        margin: 0.2rem 0;
    }

    /* ---------- Prediction result boxes ---------- */
    .pred-benign {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .pred-benign h3 { color: #065f46; margin: 0; }
    .pred-benign p { color: #047857; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

    .pred-malignant {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .pred-malignant h3 { color: #991b1b; margin: 0; }
    .pred-malignant p { color: #b91c1c; margin: 0.3rem 0 0 0; font-size: 0.9rem; }

    /* ---------- Architecture code blocks ---------- */
    .arch-block {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        line-height: 1.6;
        white-space: pre;
        overflow-x: auto;
    }

    /* ---------- Smooth dividers ---------- */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }

    /* ---------- Subtle table styling ---------- */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th {
        background: #f1f5f9 !important;
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.85rem;
    }
    td {
        font-size: 0.85rem;
        color: #475569 !important;
    }
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
        <p>AI-based skin cancer classification using Federated Learning and Hybrid CNN–Transformer models.</p>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the page footer."""
    st.markdown("""
    <div class="custom-footer">
        Final Year Project — <strong>Federated Disease Detection using Advanced AI Models</strong><br>
        Hybrid CNN + Transformer &nbsp;·&nbsp; FedAvg · FedProx · MOON &nbsp;·&nbsp; HAM10000 Dataset
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_model(model_key):
    """Load a model by its key name."""
    model_path = MODEL_PATHS.get(model_key)
    if model_path is None or not os.path.exists(model_path):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device


def predict(image, model, device):
    """Run prediction on a single PIL image."""
    img_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    prob = torch.sigmoid(output[0, 1]).item()

    if prob > 0.5:
        prediction = "Malignant"
    else:
        prediction = "Benign"

    return prediction, prob


def load_metrics(path):
    """Load metrics JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================
# Page 1 — Project Overview
# ============================================================
def page_overview():
    render_header()

    st.subheader("📋 About This Project")
    st.markdown("""
    This project implements a **Federated Learning** framework for skin cancer detection
    that enables multiple hospitals to collaboratively train a shared diagnostic model
    **without exchanging patient data**.

    The system uses the **HAM10000** dermoscopy dataset and classifies skin lesions
    into two categories: **Benign** and **Malignant**.

    Three federated optimization algorithms are implemented and compared:
    **FedAvg**, **FedProx**, and **MOON**.
    """)

    st.divider()

    # ---- Architecture (two-column) ----
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
        st.subheader("🏥 Federated Learning")
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

    # ---- Dataset ----
    st.subheader("📊 Dataset — HAM10000")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", "10,015")
    col2.metric("Benign", "8,061 (80.5%)")
    col3.metric("Malignant", "1,954 (19.5%)")

    st.markdown("""
    | Category | Original Diagnoses |
    |---|---|
    | **Malignant** | Melanoma (mel), Basal Cell Carcinoma (bcc), Actinic Keratosis (akiec) |
    | **Benign** | Melanocytic Nevi (nv), Benign Keratosis (bkl), Dermatofibroma (df), Vascular (vasc) |
    """)

    # ---- Models compared ----
    st.divider()
    st.subheader("🔄 Models Compared")
    st.markdown("""
    | # | Model | Type | Parameters |
    |---|---|---|---|
    | 1 | ResNet50 (Baseline) | Centralized CNN | ~23.5M |
    | 2 | Hybrid CNN + Transformer | Centralized Hybrid | ~28.8M |
    | 3 | Federated Hybrid (FedAvg) | Federated Learning | ~28.8M |
    | 4 | Federated Hybrid (FedProx) | Federated Learning | ~28.8M |
    | 5 | Federated Hybrid (MOON) | Federated Learning | ~28.8M |
    """)

    render_footer()


# ============================================================
# Page 2 — Image Prediction
# ============================================================
def page_prediction():
    render_header()

    st.subheader("🔍 Skin Lesion Prediction")
    st.caption(
        "Upload a dermoscopic image to classify it as Benign or Malignant."
    )

    st.divider()

    # Model selection dropdown
    available_models = []
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available_models.append(key)

    if not available_models:
        st.error(
            "No trained model files found in `weights/`. "
            "Please train at least one model first."
        )
        render_footer()
        return

    selected_model = st.selectbox(
        "🧠 Choose Model for Prediction",
        available_models,
        index=0,
    )

    st.divider()

    # Upload
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image…",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col_pred:
            with st.spinner(f"Analyzing with {selected_model}…"):
                model, device = load_model(selected_model)
                if model is None:
                    st.error(f"Model `{selected_model}` could not be loaded.")
                    render_footer()
                    return
                prediction, prob = predict(image, model, device)

            st.caption(f"**Model:** {selected_model}")

            # Prediction result — styled
            if prediction == "Benign":
                confidence = (1 - prob) * 100
                st.markdown(f"""
                <div class="pred-benign">
                    <h3>✅ Benign</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = prob * 100
                st.markdown(f"""
                <div class="pred-malignant">
                    <h3>⚠️ Malignant</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("Probability")
            st.progress(prob, text=f"Malignant: {prob*100:.1f}%")
            st.caption(f"**Malignant probability:** {prob:.4f}")

        st.divider()
        st.caption(
            "⚠️ **Disclaimer**: This tool is for research purposes only. "
            "It is NOT a substitute for professional medical diagnosis."
        )

    render_footer()


# ============================================================
# Page 3 — Model Results
# ============================================================
def page_results():
    render_header()

    st.subheader("📊 Model Performance Results")
    st.caption("Comparison of centralized and federated model performance metrics.")

    st.divider()

    # Load metrics for all models
    metrics_paths = {
        "Centralized CNN": "results/centralized/metrics.json",
        "Centralized Hybrid": "results/federated_cnn/metrics.json",
        "FedAvg Hybrid": "results/federated_hybrid/metrics.json",
        "FedProx Hybrid": "results/federated_fedprox/metrics.json",
        "MOON Hybrid": "results/federated_moon/metrics.json",
    }

    models_data = {}
    for name, path in metrics_paths.items():
        m = load_metrics(path)
        if m:
            models_data[name] = m

    if not models_data:
        st.warning(
            "No results found. Please run `python generate_results.py` first."
        )
        render_footer()
        return

    # ---- Comparison table ----
    st.subheader("📋 Performance Comparison")

    table_md = "| Metric | " + " | ".join(models_data.keys()) + " |\n"
    table_md += "|---|" + "|".join(["---"] * len(models_data)) + "|\n"

    for metric_key, metric_label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("auc_roc", "AUC-ROC"),
    ]:
        row = f"| **{metric_label}** |"
        for data in models_data.values():
            val = data.get(metric_key, 0) * 100
            row += f" {val:.2f}% |"
        table_md += row + "\n"

    st.markdown(table_md)

    st.divider()

    # ---- Comparison charts ----
    st.subheader("📈 Comparison Charts")

    col1, col2 = st.columns(2)

    if os.path.exists("results/comparison/metric_comparison.png"):
        with col1:
            st.image("results/comparison/metric_comparison.png",
                     caption="Model Performance Comparison",
                     use_container_width=True)

    if os.path.exists("results/comparison/roc_comparison.png"):
        with col2:
            st.image("results/comparison/roc_comparison.png",
                     caption="ROC Curve Comparison",
                     use_container_width=True)

    # Individual metric charts
    st.divider()
    st.subheader("📊 Individual Metric Comparisons")
    chart_cols = st.columns(3)

    chart_files = [
        ("results/plots/accuracy_comparison.png", "Accuracy Comparison"),
        ("results/plots/auc_comparison.png", "AUC-ROC Comparison"),
        ("results/plots/f1_comparison.png", "F1 Score Comparison"),
    ]

    for i, (path, label) in enumerate(chart_files):
        if os.path.exists(path):
            with chart_cols[i]:
                st.image(path, caption=label, use_container_width=True)

    if os.path.exists("results/comparison/training_curves.png"):
        st.image("results/comparison/training_curves.png",
                 caption="Training Accuracy & Loss Curves",
                 use_container_width=True)

    st.divider()

    # ---- Confusion matrices ----
    st.subheader("🔢 Confusion Matrices")

    cm_files = [
        ("results/centralized/confusion_matrix.png", "Centralized CNN"),
        ("results/federated_cnn/confusion_matrix.png", "Centralized Hybrid"),
        ("results/federated_hybrid/confusion_matrix.png", "FedAvg Hybrid"),
        ("results/federated_fedprox/confusion_matrix.png", "FedProx Hybrid"),
        ("results/federated_moon/confusion_matrix.png", "MOON Hybrid"),
    ]

    # Show in rows of 3
    for row_start in range(0, len(cm_files), 3):
        row_files = cm_files[row_start:row_start + 3]
        cm_cols = st.columns(len(row_files))
        for i, (path, label) in enumerate(row_files):
            if os.path.exists(path):
                with cm_cols[i]:
                    st.image(path, caption=label, use_container_width=True)

    render_footer()


# ============================================================
# Page 4 — Federated Learning Visualization
# ============================================================
def page_federated():
    render_header()

    st.subheader("🏥 Federated Learning Visualization")
    st.caption("Understanding the privacy-preserving federated training process.")

    st.divider()

    # ---- Training setup metrics ----
    st.subheader("📋 Federated Training Setup")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hospitals (Clients)", "3")
    col2.metric("Algorithms", "FedAvg, FedProx, MOON")
    col3.metric("Communication Rounds", "5")
    col4.metric("Local Epochs", "2")

    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Model | Hybrid CNN + Transformer |
    | Total Parameters | ~28.8M |
    | Batch Size | 32 |
    | Learning Rate | 0.0001 |
    | Optimizer | Adam |
    | FedProx μ | 0.01 |
    | MOON λ | 0.5 |
    | MOON τ | 0.5 |
    """)

    st.divider()

    # ---- Hospital data distribution ----
    st.subheader("🏥 Hospital Data Distribution (Non-IID)")

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

    st.divider()

    # ---- Convergence plot ----
    st.subheader("📈 Federated Convergence")

    if os.path.exists("results/federated_hybrid/federated_convergence.png"):
        st.image(
            "results/federated_hybrid/federated_convergence.png",
            caption="Federated Training Convergence — FedAvg vs FedProx vs MOON",
            use_container_width=True,
        )
    else:
        st.info("Convergence plot not found. Run `python generate_results.py`.")

    st.divider()

    # ---- Algorithm explanations ----
    st.subheader("🔄 Federated Optimization Algorithms")

    tab1, tab2, tab3 = st.tabs(["FedAvg", "FedProx", "MOON"])

    with tab1:
        st.markdown("**Federated Averaging (FedAvg)**")
        st.markdown("""
<div class="arch-block">For each round t = 1, 2, …, T:
  1. Server sends global model to all clients
  2. Each client trains locally for E epochs
  3. Clients send updated weights back to server
  4. Server averages all client weights:
     w_global = (1/K) × Σ w_k
  5. Updated global model is distributed</div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**Federated Proximal (FedProx)**")
        st.markdown("""
<div class="arch-block">Modifies local loss to include proximal term:
  Loss = CrossEntropyLoss + (μ/2) × ||w_local − w_global||²

  μ = 0.01 (proximal coefficient)

  This regularizes local updates to stay close
  to the global model, improving stability
  under Non-IID data distributions.</div>
        """, unsafe_allow_html=True)
        st.info("🔧 FedProx improves convergence on heterogeneous (Non-IID) data "
                "by penalizing large deviations from the global model.")

    with tab3:
        st.markdown("**Model-Contrastive Federated Learning (MOON)**")
        st.markdown("""
<div class="arch-block">Adds contrastive loss on model embeddings:
  Loss = CrossEntropyLoss + λ × ContrastiveLoss

  Positive pair: (local embedding, global embedding)
  Negative pair: (local embedding, previous local embedding)

  λ = 0.5 (contrastive weight)
  τ = 0.5 (temperature)

  Improves representation consistency between
  local and global models.</div>
        """, unsafe_allow_html=True)
        st.info("🧠 MOON leverages contrastive learning to align local model "
                "representations with the global model, reducing client drift.")

    st.markdown("")
    st.info("🔒 **Key Advantage:** Patient data **never leaves** the hospital. "
            "Only model weight updates are communicated.")

    # ---- Results ----
    st.divider()
    st.subheader("🏆 Federated Model Results")

    fed_metrics = {
        "FedAvg": load_metrics("results/federated_hybrid/metrics.json"),
        "FedProx": load_metrics("results/federated_fedprox/metrics.json"),
        "MOON": load_metrics("results/federated_moon/metrics.json"),
    }

    for algo_name, metrics in fed_metrics.items():
        if metrics:
            st.markdown(f"**{algo_name}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
            c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
            c4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
            c5.metric("AUC-ROC", f"{metrics['auc_roc']*100:.1f}%")
            st.markdown("")

    render_footer()


# ============================================================
# Main — Navigation
# ============================================================
def main():
    st.set_page_config(
        page_title="Federated Skin Cancer Detection System",
        page_icon="🔬",
        layout="wide",
    )

    # Inject custom styles
    inject_custom_css()

    # ---- Sidebar ----
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
            "Project Overview",
            "Image Prediction",
            "Model Results",
            "Federated Learning",
        ],
    )

    st.sidebar.divider()

    # Sidebar info
    st.sidebar.markdown("""
    **Tech Stack**
    - 🧠 PyTorch
    - 🏗️ ResNet-50 + Transformer
    - 🔄 FedAvg · FedProx · MOON
    - 📊 HAM10000 Dataset
    """)

    st.sidebar.divider()
    st.sidebar.caption("© 2026 · Final Year Project")

    # ---- Route to page ----
    if page == "Project Overview":
        page_overview()
    elif page == "Image Prediction":
        page_prediction()
    elif page == "Model Results":
        page_results()
    elif page == "Federated Learning":
        page_federated()


if __name__ == "__main__":
    main()

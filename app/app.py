import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CerviImagingDiag — Federated AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. Global CSS  (injected once via st.markdown — only <style> tag, always safe)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%,  rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(0,212,255,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%,  rgba(168,85,247,0.06) 0%, transparent 70%),
        linear-gradient(160deg, #060912 0%, #0b1120 35%, #080f1c 65%, #0a0e18 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem !important; max-width: 1120px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse 120% 60% at 50% 0%, rgba(99,102,241,0.2) 0%, transparent 60%),
        linear-gradient(180deg, #0b1220 0%, #080e1a 60%, #060c16 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.25) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"] .block-container { padding: 0 1.2rem 2rem 1.2rem !important; }

/* ── Patient ID Label ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label {
    color: #a5b4fc !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── Patient ID Input Box ── */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] .stTextInput input {
    background: linear-gradient(135deg, rgba(30,40,80,0.97) 0%, rgba(20,30,65,0.97) 100%) !important;
    border: 1.5px solid rgba(99,102,241,0.7) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 0.9rem !important;
    caret-color: #00d4ff !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] input[type="text"]:focus,
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.22), 0 0 20px rgba(0,212,255,0.18) !important;
    background: linear-gradient(135deg, rgba(35,50,100,0.99) 0%, rgba(25,38,85,0.99) 100%) !important;
}
[data-testid="stSidebar"] input[type="text"]::placeholder {
    color: rgba(160,180,230,0.45) !important;
    font-weight: 400 !important;
}

/* ── Run Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #00d4ff 50%, #a855f7 100%) !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.4), 0 0 0 1px rgba(255,255,255,0.08) !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.55), 0 0 60px rgba(0,212,255,0.25) !important;
}

/* ── Download Button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(99,102,241,0.1) 100%) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.45) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.6rem !important;
    width: auto !important;
    transition: all 0.3s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.18) 0%, rgba(99,102,241,0.18) 100%) !important;
    border-color: #00d4ff !important;
    box-shadow: 0 0 24px rgba(0,212,255,0.25) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] section {
    background: rgba(0,212,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00d4ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060912; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.5); }

/* ── Streamlit st.image ── */
[data-testid="stImage"] img { border-radius: 10px; }

/* ── st.metric overrides ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(12,20,42,0.97) 0%, rgba(8,15,34,0.97) 100%);
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 16px;
    padding: 1.2rem 1.4rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: rgba(0,212,255,0.65) !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #e0f8ff !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00d4ff, #6366f1, #a855f7) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
    height: 8px !important;
}

/* ── Columns gap ── */
[data-testid="stHorizontalBlock"] { gap: 1.25rem; }

/* ── st.success ── */
.stAlert[data-baseweb="notification"] {
    border-radius: 12px !important;
}
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}

@keyframes blink {
    0%,100% { opacity:1; box-shadow: 0 0 8px #00d4ff; }
    50%      { opacity:0.4; box-shadow: 0 0 3px #00d4ff; }
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper: render HTML safely using st.html()
# ─────────────────────────────────────────────
def H(html: str):
    """Render arbitrary HTML using the modern st.html() API."""
    st.html(html)

# ─────────────────────────────────────────────
# 3. Class Configuration
# ─────────────────────────────────────────────
CLASSES = [
    'im_Dyskeratotic',
    'im_Koilocytotic',
    'im_Metaplastic',
    'im_Parabasal',
    'im_Superficial-Intermediate'
]

CLASS_MEDICAL_PROFILES = {
    'im_Dyskeratotic': {
        'grade': 'Abnormal / High-Grade Squamous Intraepithelial Precursor',
        'severity': 'high',
        'icon': '🔴',
        'color': '#f87171',
        'badge': '<span style="display:inline-flex;align-items:center;gap:0.35rem;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.45);color:#f87171;padding:0.28rem 0.75rem;border-radius:20px;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">⚠ High Risk</span>',
        'body_bg': 'rgba(239,68,68,0.07)',
        'body_border': '#f87171',
        'description': 'Shows distinct cellular keratinization and nuclear atypia.',
        'clinical_advice': 'Indicates potential high-grade precancerous changes. Immediate consultation with a gynecological oncologist and a colposcopy-guided biopsy are strongly recommended.'
    },
    'im_Koilocytotic': {
        'grade': 'Abnormal / Low-Grade Precancerous Change (HPV-related)',
        'severity': 'medium',
        'icon': '🟠',
        'color': '#fb923c',
        'badge': '<span style="display:inline-flex;align-items:center;gap:0.35rem;background:rgba(251,146,60,0.15);border:1px solid rgba(251,146,60,0.45);color:#fb923c;padding:0.28rem 0.75rem;border-radius:20px;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">◎ Moderate Risk</span>',
        'body_bg': 'rgba(251,146,60,0.07)',
        'body_border': '#fb923c',
        'description': 'Exhibits characteristic koilocytotic halos, indicative of Human Papillomavirus (HPV) infection and mild dysplasia.',
        'clinical_advice': 'Low-grade lesion associated with viral changes. Routine follow-up screening within 6 months and further HPV-typing tests are advised.'
    },
    'im_Metaplastic': {
        'grade': 'Benign / Metaplastic Transformation',
        'severity': 'normal',
        'icon': '🟢',
        'color': '#4ade80',
        'badge': '<span style="display:inline-flex;align-items:center;gap:0.35rem;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.45);color:#4ade80;padding:0.28rem 0.75rem;border-radius:20px;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">✓ Normal</span>',
        'body_bg': 'rgba(34,197,94,0.07)',
        'body_border': '#4ade80',
        'description': 'Represents normal tissue healing or cellular adaptation within the transformation zone of the cervix.',
        'clinical_advice': 'Benign finding with favorable prognosis. Standard annual routine cervical screening is recommended.'
    },
    'im_Parabasal': {
        'grade': 'Benign / Immature Squamous Cells',
        'severity': 'normal',
        'icon': '🟢',
        'color': '#4ade80',
        'badge': '<span style="display:inline-flex;align-items:center;gap:0.35rem;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.45);color:#4ade80;padding:0.28rem 0.75rem;border-radius:20px;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">✓ Normal</span>',
        'body_bg': 'rgba(34,197,94,0.07)',
        'body_border': '#4ade80',
        'description': 'Normal deep epithelial cells occasionally observed in atrophic or regular physiological smears.',
        'clinical_advice': 'Normal physiological presentation. No immediate clinical intervention required.'
    },
    'im_Superficial-Intermediate': {
        'grade': 'Normal / Healthy Mature Cells',
        'severity': 'normal',
        'icon': '🟢',
        'color': '#4ade80',
        'badge': '<span style="display:inline-flex;align-items:center;gap:0.35rem;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.45);color:#4ade80;padding:0.28rem 0.75rem;border-radius:20px;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">✓ Normal</span>',
        'body_bg': 'rgba(34,197,94,0.07)',
        'body_border': '#4ade80',
        'description': 'Normal mature squamous epithelial cells reflecting healthy cellular shedding.',
        'clinical_advice': 'Normal finding. Maintain regular age-appropriate preventive screenings.'
    }
}

# ─────────────────────────────────────────────
# 4. Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    H("""
    <div style="text-align:center;padding:2rem 0 1.5rem 0;">
        <div style="width:60px;height:60px;margin:0 auto 0.8rem auto;
                    background:linear-gradient(135deg,#6366f1 0%,#00d4ff 100%);
                    border-radius:16px;display:flex;align-items:center;
                    justify-content:center;font-size:1.8rem;
                    box-shadow:0 8px 32px rgba(99,102,241,0.5),0 0 0 1px rgba(255,255,255,0.08);">
            🔬
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:800;
                    background:linear-gradient(135deg,#a5b4fc 0%,#00d4ff 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">CerviImagingDiag</div>
        <div style="font-size:0.65rem;color:rgba(180,190,220,0.45);
                    letter-spacing:0.12em;text-transform:uppercase;margin-top:0.2rem;">
            Federated AI Platform v2.0
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.4),rgba(0,212,255,0.3),transparent);margin-bottom:1.5rem;"></div>
    """)

    patient_id = st.text_input(
        "PATIENT ID",
        value="P-2026-01",
        placeholder="e.g. P-2026-01"
    )

    H("""
    <div style="height:1.2rem;"></div>
    <div style="border-radius:12px;padding:1rem 1.1rem;margin-bottom:0.85rem;
                background:linear-gradient(135deg,rgba(99,102,241,0.18) 0%,rgba(99,102,241,0.08) 100%);
                border:1px solid rgba(99,102,241,0.35);">
        <div style="font-weight:700;font-size:0.75rem;color:#a5b4fc;margin-bottom:0.4rem;">🛡 Privacy-Preserving</div>
        <div style="color:rgba(180,200,240,0.75);font-size:0.73rem;line-height:1.55;">
            Model trained using Federated Learning — no raw patient data leaves local nodes during training.
        </div>
    </div>
    <div style="border-radius:12px;padding:1rem 1.1rem;margin-bottom:0.85rem;
                background:linear-gradient(135deg,rgba(0,212,255,0.12) 0%,rgba(0,212,255,0.05) 100%);
                border:1px solid rgba(0,212,255,0.28);">
        <div style="font-weight:700;font-size:0.75rem;color:#00d4ff;margin-bottom:0.4rem;">🤖 Model Architecture</div>
        <div style="color:rgba(180,200,240,0.75);font-size:0.73rem;line-height:1.55;">
            MobileNetV3-Small · 5-class SIPaKMeD · FL Round 5
        </div>
    </div>
    <div style="border-radius:12px;padding:1rem 1.1rem;margin-bottom:0.85rem;
                background:linear-gradient(135deg,rgba(16,185,129,0.14) 0%,rgba(16,185,129,0.06) 100%);
                border:1px solid rgba(16,185,129,0.3);">
        <div style="font-weight:700;font-size:0.75rem;color:#4ade80;margin-bottom:0.4rem;">📊 Performance</div>
        <div style="color:rgba(180,200,240,0.75);font-size:0.73rem;line-height:1.55;">
            FL accuracy: <strong style="color:#4ade80;">98.21%</strong><br>
            Centralized baseline: <strong style="color:#a5b4fc;">93.62%</strong>
        </div>
    </div>
    <div style="margin-top:1rem;padding:0.8rem 1rem;
                background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.18);
                border-radius:10px;font-size:0.68rem;color:rgba(210,160,160,0.7);
                line-height:1.55;text-align:center;">
        ⚠ For research &amp; educational use only.<br>Not a substitute for clinical diagnosis.
    </div>
    """)

# ─────────────────────────────────────────────
# 5. Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    try:
        model.load_state_dict(torch.load(
            "federated_cervical_model.pth", map_location=device, weights_only=True
        ))
    except FileNotFoundError:
        st.error("❌  Model checkpoint not found. Launch from the `app/` directory.")
        st.stop()
    model.to(device)
    model.eval()
    return model, device

model, device = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# 6. Hero Section
# ─────────────────────────────────────────────
H("""
<div style="position:relative;
            background:radial-gradient(ellipse 100% 100% at 100% 0%,rgba(99,102,241,0.22) 0%,transparent 55%),
                       radial-gradient(ellipse 80% 80% at 0% 100%,rgba(0,212,255,0.14) 0%,transparent 55%),
                       linear-gradient(135deg,rgba(15,22,45,0.97) 0%,rgba(10,16,35,0.97) 100%);
            border:1px solid rgba(99,102,241,0.3);border-radius:24px;
            padding:2.5rem 3rem 2rem 3rem;margin-bottom:1.75rem;overflow:hidden;">

  <!-- top gradient bar -->
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,#6366f1 0%,#00d4ff 50%,#a855f7 100%);
              border-radius:24px 24px 0 0;"></div>

  <!-- grid overlay -->
  <div style="position:absolute;inset:0;
              background-image:linear-gradient(rgba(99,102,241,0.06) 1px,transparent 1px),
                               linear-gradient(90deg,rgba(99,102,241,0.06) 1px,transparent 1px);
              background-size:40px 40px;
              mask-image:radial-gradient(ellipse 80% 80% at 100% 0%,black 20%,transparent 70%);
              pointer-events:none;"></div>

  <!-- live badge -->
  <div style="display:inline-flex;align-items:center;gap:0.45rem;
              background:linear-gradient(135deg,rgba(0,212,255,0.15) 0%,rgba(99,102,241,0.12) 100%);
              border:1px solid rgba(0,212,255,0.4);border-radius:20px;
              padding:0.3rem 1rem;font-size:0.68rem;font-weight:700;
              color:#00d4ff;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1.2rem;">
    <span style="width:7px;height:7px;background:#00d4ff;border-radius:50%;
                 box-shadow:0 0 8px #00d4ff;animation:blink 2s infinite;display:inline-block;"></span>
    Live · Federated AI Active
  </div>

  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:800;
             line-height:1.15;background:linear-gradient(135deg,#ffffff 0%,#c7d8ff 35%,#00d4ff 70%,#a855f7 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin:0 0 0.9rem 0;">
    Cervical Cell AI<br>Diagnostics Platform
  </h1>

  <p style="color:rgba(180,210,240,0.7);font-size:0.93rem;line-height:1.65;
            max-width:580px;margin:0 0 2rem 0;">
    A privacy-preserving federated learning framework for early-stage cervical cytology classification.
    Upload a cell image for instant AI-powered clinical analysis.
  </p>

  <div style="display:flex;gap:0;background:rgba(255,255,255,0.03);
              border:1px solid rgba(255,255,255,0.08);border-radius:16px;
              overflow:hidden;width:fit-content;">
    <div style="padding:1rem 1.8rem;text-align:center;border-right:1px solid rgba(255,255,255,0.07);">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4ff,#6366f1);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">98.21%</div>
      <div style="font-size:0.65rem;color:rgba(160,190,230,0.55);text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">FL Accuracy</div>
    </div>
    <div style="padding:1rem 1.8rem;text-align:center;border-right:1px solid rgba(255,255,255,0.07);">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4ff,#6366f1);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">5</div>
      <div style="font-size:0.65rem;color:rgba(160,190,230,0.55);text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">Cell Classes</div>
    </div>
    <div style="padding:1rem 1.8rem;text-align:center;border-right:1px solid rgba(255,255,255,0.07);">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4ff,#6366f1);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">3</div>
      <div style="font-size:0.65rem;color:rgba(160,190,230,0.55);text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">FL Clients</div>
    </div>
    <div style="padding:1rem 1.8rem;text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4ff,#6366f1);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">5</div>
      <div style="font-size:0.65rem;color:rgba(160,190,230,0.55);text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">FL Rounds</div>
    </div>
  </div>
</div>
""")

# ─────────────────────────────────────────────
# 7. Upload Section
# ─────────────────────────────────────────────
H("""
<div style="position:relative;
            background:radial-gradient(ellipse 60% 80% at 0% 50%,rgba(0,212,255,0.07) 0%,transparent 60%),
                       linear-gradient(135deg,rgba(12,20,40,0.97) 0%,rgba(8,14,30,0.97) 100%);
            border:1.5px dashed rgba(0,212,255,0.35);border-radius:20px;
            padding:1.6rem 2rem 1.2rem 2rem;margin-bottom:0.5rem;">
  <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.4rem;">
    <div style="width:38px;height:38px;
                background:linear-gradient(135deg,rgba(0,212,255,0.2) 0%,rgba(99,102,241,0.2) 100%);
                border:1px solid rgba(0,212,255,0.3);border-radius:10px;
                display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📂</div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:700;color:#e8f4ff;">
      Upload Cervical Cell Image
    </div>
  </div>
  <div style="color:rgba(140,175,215,0.55);font-size:0.76rem;margin-bottom:0.8rem;padding-left:50px;">
    Supported formats: BMP · JPG · PNG &nbsp;|&nbsp; Max 200MB per file
  </div>
</div>
""")

uploaded_file = st.file_uploader(
    "Drop your cell image here",
    type=["bmp", "jpg", "png"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# 8. Preview + Run
# ─────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_det = st.columns([3, 2], gap="large")

    with col_img:
        H("""<div style="background:linear-gradient(135deg,rgba(12,20,42,0.98) 0%,rgba(8,15,32,0.98) 100%);
                          border:1px solid rgba(99,102,241,0.25);border-radius:18px;overflow:hidden;
                          box-shadow:0 8px 40px rgba(0,0,0,0.4);margin-bottom:0.25rem;">
               <div style="padding:0.75rem 1.2rem;
                           background:linear-gradient(90deg,rgba(99,102,241,0.15) 0%,rgba(0,212,255,0.08) 100%);
                           border-bottom:1px solid rgba(99,102,241,0.2);
                           font-size:0.72rem;color:#a5b4fc;text-transform:uppercase;
                           letter-spacing:0.12em;font-weight:600;">
                 🔬 Cytology Sample Preview
               </div>""")
        st.image(image, use_container_width=True)
        H("</div>")

    with col_det:
        H(f"""
        <div style="border-radius:18px;padding:1.5rem;height:100%;position:relative;overflow:hidden;
                    background:radial-gradient(ellipse 80% 80% at 100% 0%,rgba(168,85,247,0.15) 0%,transparent 55%),
                               linear-gradient(160deg,rgba(15,22,50,0.98) 0%,rgba(10,15,38,0.98) 100%);
                    border:1px solid rgba(168,85,247,0.25);box-shadow:0 8px 40px rgba(0,0,0,0.35);">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;
                      background:linear-gradient(90deg,#a855f7,#6366f1,#00d4ff);"></div>
          <div style="font-size:0.68rem;color:rgba(200,180,255,0.5);text-transform:uppercase;
                      letter-spacing:0.14em;font-weight:600;margin-bottom:1.4rem;">Sample Details</div>

          <div style="margin-bottom:1.1rem;">
            <div style="font-size:0.68rem;color:rgba(180,190,240,0.45);text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.3rem;">Patient ID</div>
            <div style="font-size:0.9rem;color:#00d4ff;font-weight:700;">{patient_id}</div>
          </div>
          <div style="margin-bottom:1.1rem;">
            <div style="font-size:0.68rem;color:rgba(180,190,240,0.45);text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.3rem;">File Name</div>
            <div style="font-size:0.82rem;color:#e8f0ff;font-weight:500;word-break:break-all;">{uploaded_file.name}</div>
          </div>
          <div style="margin-bottom:1.1rem;">
            <div style="font-size:0.68rem;color:rgba(180,190,240,0.45);text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.3rem;">Dimensions</div>
            <div style="font-size:0.9rem;color:#e8f0ff;font-weight:600;">{image.width} × {image.height} px</div>
          </div>
          <div style="margin-bottom:1.1rem;">
            <div style="font-size:0.68rem;color:rgba(180,190,240,0.45);text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.3rem;">Model</div>
            <div style="font-size:0.88rem;color:#c084fc;font-weight:600;">MobileNetV3-Small · Federated</div>
          </div>
          <div style="padding-top:1rem;border-top:1px solid rgba(168,85,247,0.15);margin-top:0.5rem;">
            <div style="font-size:0.72rem;color:rgba(190,170,255,0.5);line-height:1.55;">
              ✅ Image loaded &amp; ready for analysis.<br>Press the button below to classify.
            </div>
          </div>
        </div>
        """)

    H("<div style='height:0.75rem;'></div>")
    run_btn = st.button("🚀  Run Federated AI Diagnostics", use_container_width=True)

    # ─────────────────────────────────────────────
    # 9. Results
    # ─────────────────────────────────────────────
    if run_btn:
        with st.spinner("Analyzing across federated nodes…"):
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

        pred_class  = CLASSES[predicted_idx.item()]
        conf_score  = confidence.item() * 100
        profile     = CLASS_MEDICAL_PROFILES[pred_class]
        sev_color   = profile['color']
        badge_html  = profile['badge']
        all_probs   = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

        # Confidence bar gradient
        conf_gradients = {
            'high':   f'linear-gradient(90deg,#f87171,#ef4444)',
            'medium': f'linear-gradient(90deg,#fb923c,#f97316)',
            'normal': f'linear-gradient(90deg,#4ade80,#22c55e)',
        }
        bar_grad = conf_gradients[profile['severity']]

        # ── Results header ──
        H("""
        <div style="display:flex;align-items:center;gap:0.9rem;margin:2rem 0 1.2rem 0;">
          <div style="width:4px;height:32px;background:linear-gradient(180deg,#00d4ff,#6366f1,#a855f7);
                      border-radius:2px;box-shadow:0 0 12px rgba(0,212,255,0.5);"></div>
          <span style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;
                       font-weight:800;color:#e8f4ff;">Diagnostic Results</span>
        </div>
        """)

        # ── Three metric cards ──
        mc1, mc2, mc3 = st.columns(3, gap="medium")

        with mc1:
            H(f"""
            <div style="border-radius:18px;padding:1.5rem 1.4rem;position:relative;overflow:hidden;
                        background:linear-gradient(135deg,rgba(0,212,255,0.14) 0%,rgba(0,180,220,0.06) 100%);
                        border:1px solid rgba(0,212,255,0.35);
                        box-shadow:0 8px 32px rgba(0,212,255,0.1),inset 0 1px 0 rgba(0,212,255,0.1);
                        transition:transform 0.3s;">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,#00d4ff,#00a0c4);"></div>
              <div style="width:36px;height:36px;border-radius:10px;
                          background:rgba(0,212,255,0.15);border:1px solid rgba(0,212,255,0.25);
                          display:flex;align-items:center;justify-content:center;
                          font-size:1.1rem;margin-bottom:1rem;">🧬</div>
              <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;
                          font-weight:600;color:rgba(0,212,255,0.65);margin-bottom:0.5rem;">Predicted Class</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;
                          font-weight:800;color:#e0f8ff;line-height:1.2;">{pred_class}</div>
              {badge_html}
            </div>
            """)

        with mc2:
            H(f"""
            <div style="border-radius:18px;padding:1.5rem 1.4rem;position:relative;overflow:hidden;
                        background:linear-gradient(135deg,rgba(99,102,241,0.18) 0%,rgba(80,85,220,0.07) 100%);
                        border:1px solid rgba(99,102,241,0.4);
                        box-shadow:0 8px 32px rgba(99,102,241,0.12),inset 0 1px 0 rgba(99,102,241,0.12);">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,#6366f1,#818cf8);"></div>
              <div style="width:36px;height:36px;border-radius:10px;
                          background:rgba(99,102,241,0.18);border:1px solid rgba(99,102,241,0.3);
                          display:flex;align-items:center;justify-content:center;
                          font-size:1.1rem;margin-bottom:1rem;">📊</div>
              <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;
                          font-weight:600;color:rgba(130,140,255,0.65);margin-bottom:0.5rem;">Confidence Score</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;
                          font-weight:800;color:{sev_color};">{conf_score:.2f}%</div>
              <div style="margin-top:0.9rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.68rem;
                            color:rgba(160,200,230,0.5);margin-bottom:0.5rem;">
                  <span>0%</span><span>100%</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.07);border-radius:4px;overflow:hidden;">
                  <div style="width:{conf_score:.1f}%;height:100%;border-radius:4px;
                              background:{bar_grad};box-shadow:0 0 10px {sev_color}66;"></div>
                </div>
              </div>
            </div>
            """)

        with mc3:
            H(f"""
            <div style="border-radius:18px;padding:1.5rem 1.4rem;position:relative;overflow:hidden;
                        background:linear-gradient(135deg,rgba(168,85,247,0.16) 0%,rgba(140,60,210,0.07) 100%);
                        border:1px solid rgba(168,85,247,0.38);
                        box-shadow:0 8px 32px rgba(168,85,247,0.1),inset 0 1px 0 rgba(168,85,247,0.1);">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,#a855f7,#c084fc);"></div>
              <div style="width:36px;height:36px;border-radius:10px;
                          background:rgba(168,85,247,0.16);border:1px solid rgba(168,85,247,0.28);
                          display:flex;align-items:center;justify-content:center;
                          font-size:1.1rem;margin-bottom:1rem;">{profile['icon']}</div>
              <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;
                          font-weight:600;color:rgba(196,132,252,0.65);margin-bottom:0.5rem;">Diagnostic Category</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;
                          font-weight:700;color:#f3e8ff;line-height:1.4;">{profile['grade']}</div>
            </div>
            """)

        # ── Probability Distribution ──
        prob_rows_html = ""
        for cls, prob in zip(CLASSES, all_probs):
            pct = prob * 100
            is_pred = cls == pred_class
            if is_pred:
                bar_s  = "background:linear-gradient(90deg,#00d4ff,#6366f1,#a855f7);box-shadow:0 0 12px rgba(0,212,255,0.4);"
                name_s = "color:#00d4ff;font-weight:700;"
                val_s  = "color:#00d4ff;font-weight:800;"
            else:
                bar_s  = "background:rgba(99,102,241,0.2);"
                name_s = "color:rgba(160,190,240,0.55);font-weight:400;"
                val_s  = "color:rgba(140,170,220,0.55);font-weight:600;"
            prob_rows_html += f"""
            <div style="display:flex;align-items:center;gap:0.9rem;margin-bottom:0.75rem;">
              <div style="width:210px;font-size:0.77rem;white-space:nowrap;
                          overflow:hidden;text-overflow:ellipsis;flex-shrink:0;{name_s}">{cls}</div>
              <div style="flex:1;height:9px;background:rgba(255,255,255,0.055);border-radius:5px;overflow:hidden;">
                <div style="width:{pct:.2f}%;height:100%;border-radius:5px;{bar_s}"></div>
              </div>
              <div style="width:48px;text-align:right;font-size:0.8rem;{val_s}">{pct:.1f}%</div>
            </div>"""

        H(f"""
        <div style="position:relative;overflow:hidden;
                    background:radial-gradient(ellipse 60% 60% at 100% 100%,rgba(99,102,241,0.1) 0%,transparent 55%),
                               linear-gradient(135deg,rgba(12,20,42,0.98) 0%,rgba(8,14,32,0.98) 100%);
                    border:1px solid rgba(99,102,241,0.22);border-radius:20px;
                    padding:1.5rem 1.8rem;margin-top:1.5rem;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;
                      background:linear-gradient(90deg,#6366f1,#00d4ff,#a855f7);"></div>
          <div style="font-size:0.7rem;color:rgba(160,190,240,0.5);text-transform:uppercase;
                      letter-spacing:0.12em;font-weight:600;margin-bottom:1.3rem;">
            📈 Class Probability Distribution
          </div>
          {prob_rows_html}
        </div>
        """)

        # ── Clinical Report ──
        H(f"""
        <div style="position:relative;overflow:hidden;
                    background:radial-gradient(ellipse 80% 50% at 0% 0%,rgba(0,212,255,0.09) 0%,transparent 50%),
                               radial-gradient(ellipse 80% 50% at 100% 100%,rgba(168,85,247,0.09) 0%,transparent 50%),
                               linear-gradient(160deg,rgba(12,20,45,0.99) 0%,rgba(8,14,34,0.99) 100%);
                    border:1px solid rgba(0,212,255,0.2);border-radius:24px;
                    padding:2rem 2.5rem 1.8rem 2.5rem;margin-top:1.8rem;
                    box-shadow:0 16px 60px rgba(0,0,0,0.45),inset 0 1px 0 rgba(255,255,255,0.04);">

          <!-- animated shimmer top bar -->
          <div style="position:absolute;top:0;left:0;right:0;height:3px;
                      background:linear-gradient(90deg,#00d4ff 0%,#6366f1 35%,#a855f7 65%,#00d4ff 100%);
                      background-size:200% auto;
                      animation:shimmer 4s linear infinite;border-radius:24px 24px 0 0;"></div>

          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:800;
                      color:#e8f4ff;margin-bottom:1.5rem;display:flex;align-items:center;gap:0.6rem;">
            📋 AI-Generated Clinical Report
          </div>

          <!-- Meta row -->
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.9rem;
                      margin-bottom:1.8rem;padding-bottom:1.5rem;
                      border-bottom:1px solid rgba(255,255,255,0.06);">
            <div>
              <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;
                          color:rgba(160,190,240,0.45);margin-bottom:0.3rem;">Patient ID</div>
              <div style="font-size:0.88rem;font-weight:600;color:#00d4ff;">{patient_id}</div>
            </div>
            <div>
              <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;
                          color:rgba(160,190,240,0.45);margin-bottom:0.3rem;">Classification</div>
              <div style="font-size:0.82rem;font-weight:600;color:#e0ecff;">{pred_class}</div>
            </div>
            <div>
              <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;
                          color:rgba(160,190,240,0.45);margin-bottom:0.3rem;">Confidence</div>
              <div style="font-size:0.88rem;font-weight:700;color:{sev_color};">{conf_score:.2f}%</div>
            </div>
            <div>
              <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;
                          color:rgba(160,190,240,0.45);margin-bottom:0.3rem;">Risk Level</div>
              <div>{badge_html}</div>
            </div>
          </div>

          <!-- Section 01 -->
          <div style="margin-bottom:1.4rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;">
              <div style="width:24px;height:24px;background:linear-gradient(135deg,#6366f1,#00d4ff);
                          border-radius:6px;display:flex;align-items:center;justify-content:center;
                          font-size:0.65rem;font-weight:800;color:white;flex-shrink:0;">01</div>
              <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.1em;color:#00d4ff;">Summary of Findings</div>
              <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(99,102,241,0.25),transparent);"></div>
            </div>
            <div style="font-size:0.87rem;line-height:1.72;padding:0.9rem 1.1rem 0.9rem 1.3rem;
                        border-radius:0 12px 12px 12px;border-left:3px solid {sev_color};
                        background:{profile['body_bg']};color:rgba(190,215,245,0.88);">
              The federated edge model detected features consistent with
              <strong style="color:{sev_color};">{profile['grade']}</strong>
              with a model confidence of <strong>{conf_score:.2f}%</strong>.
              This classification is derived from a MobileNetV3-Small model trained
              across 3 federated clients over 5 rounds.
            </div>
          </div>

          <!-- Section 02 -->
          <div style="margin-bottom:1.4rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;">
              <div style="width:24px;height:24px;background:linear-gradient(135deg,#6366f1,#00d4ff);
                          border-radius:6px;display:flex;align-items:center;justify-content:center;
                          font-size:0.65rem;font-weight:800;color:white;flex-shrink:0;">02</div>
              <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.1em;color:#a5b4fc;">Clinical Interpretation</div>
              <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(165,180,252,0.25),transparent);"></div>
            </div>
            <div style="font-size:0.87rem;line-height:1.72;padding:0.9rem 1.1rem 0.9rem 1.3rem;
                        border-radius:0 12px 12px 12px;border-left:3px solid #6366f1;
                        background:rgba(99,102,241,0.07);color:rgba(190,215,245,0.88);">
              <em>Cellular Observations:</em> {profile['description']}
            </div>
          </div>

          <!-- Section 03 -->
          <div style="margin-bottom:0.5rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;">
              <div style="width:24px;height:24px;background:linear-gradient(135deg,#6366f1,#00d4ff);
                          border-radius:6px;display:flex;align-items:center;justify-content:center;
                          font-size:0.65rem;font-weight:800;color:white;flex-shrink:0;">03</div>
              <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.1em;color:#c084fc;">Recommended Next Steps</div>
              <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(192,132,252,0.25),transparent);"></div>
            </div>
            <div style="font-size:0.87rem;line-height:1.72;padding:0.9rem 1.1rem 0.9rem 1.3rem;
                        border-radius:0 12px 12px 12px;border-left:3px solid {sev_color};
                        background:{profile['body_bg']};color:rgba(190,215,245,0.88);">
              {profile['clinical_advice']}
            </div>
          </div>

        </div>
        """)

        # ── Download ──
        report_text = f"""CerviImagingDiag — AI Clinical Report
======================================
Patient ID          : {patient_id}
Classification      : {pred_class}
Diagnostic Category : {profile['grade']}
Confidence Score    : {conf_score:.2f}%
Risk Level          : {profile['severity'].upper()}

1. SUMMARY OF FINDINGS
-----------------------
The federated edge model detected features matching "{profile['grade']}"
with a confidence of {conf_score:.2f}%.

2. CLINICAL INTERPRETATION
---------------------------
Cellular Observations: {profile['description']}

3. RECOMMENDED NEXT STEPS
--------------------------
{profile['clinical_advice']}

======================================
DISCLAIMER: AI-generated. Not a substitute for professional medical advice.
"""
        H("<div style='height:1rem;'></div>")
        st.download_button(
            label="⬇  Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"CerviDiag_{patient_id}.txt",
            mime="text/plain"
        )

        H("""
        <div style="margin-top:1.5rem;padding:0.9rem 1.3rem;
                    background:linear-gradient(135deg,rgba(239,68,68,0.07) 0%,rgba(251,146,60,0.04) 100%);
                    border:1px solid rgba(239,68,68,0.22);border-radius:12px;
                    font-size:0.76rem;color:rgba(220,170,170,0.8);line-height:1.6;">
          <strong style="color:#f87171;">⚠ Medical Disclaimer:</strong>
          This AI analysis is provided for research and educational purposes only.
          It is <strong>not</strong> intended to replace professional medical advice, diagnosis, or treatment.
          Always seek the guidance of a qualified, licensed healthcare provider.
        </div>
        """)
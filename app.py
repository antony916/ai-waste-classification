from pathlib import Path
import time

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH
from src.model import build_model

ROOT = Path(__file__).resolve().parent

theme_mode = getattr(st.context.theme, "type", "light") or "light"

st.set_page_config(
    page_title="Waste AI | Smart Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


if theme_mode == "dark":
    st.markdown("""<style>:root{--app-bg:#0D1713;--surface:#16231D;--surface-soft:#183A2A;--text:#ECF7F0;--muted:#AFC2B7;--border:#2D4539;--accent:#49C982;--accent-dark:#8BE0B2;--accent-soft:#214C36;--shadow:rgba(0,0,0,.24);--warning-bg:#332A16;--warning-border:#66552C;--warning-text:#F3D98D;}</style>""", unsafe_allow_html=True)

st.markdown(
    """
<style>
:root {
    --app-bg: #F5F8F6;
    --surface: #FFFFFF;
    --surface-soft: #EEF8F2;
    --text: #173226;
    --muted: #667970;
    --border: #DCE9E1;
    --accent: #198754;
    --accent-dark: #0F4D31;
    --accent-soft: #E6F5EC;
    --shadow: rgba(26,60,44,.07);
    --warning-bg: #FFF8E8;
    --warning-border: #F2DFAD;
    --warning-text: #66501F;
}
.stApp { background: var(--app-bg); color: var(--text); }
.block-container { max-width: 1240px; padding: 1.75rem 2rem 4rem; }
[data-testid="stHeader"] { background: var(--app-bg); }
[data-testid="stSidebar"] { background: ${"#09110E" if theme_mode == "dark" else "#102F24"}; border-right: 0; }
[data-testid="stSidebar"] * { color: #EDF8F2 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }

.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg,#0D3A29 0%,#126B45 55%,#1A8C5A 100%);
    border-radius: 28px; padding: 2.4rem 2.7rem; color: white;
    margin-bottom: 1.35rem; box-shadow: 0 18px 45px rgba(14,69,46,.18);
}
.hero:after { content: "♻"; position: absolute; right: 2rem; top: -.9rem; font-size: 9rem; opacity: .08; }
.hero-kicker {
    display: inline-block; background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.2); border-radius: 999px;
    padding: .38rem .75rem; font-size: .78rem; font-weight: 750;
    letter-spacing: .08em; text-transform: uppercase;
}
.hero h1 { margin: .9rem 0 .45rem; font-size: clamp(2rem,4vw,2.65rem); line-height: 1.08; font-weight: 800; }
.hero p { max-width: 720px; margin: 0; font-size: 1.03rem; line-height: 1.55; color: rgba(255,255,255,.9); }

.section-title { font-size: 1.25rem; font-weight: 800; color: var(--text); margin: 1.45rem 0 .55rem; }
.section-subtitle { color: var(--muted); font-size: .92rem; line-height: 1.5; margin-bottom: .9rem; }

.card,.stat-card,.step-card,.result-card { box-sizing: border-box; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 1.2rem 1.3rem; box-shadow: 0 8px 25px var(--shadow); }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.15rem; min-height: 96px; box-shadow: 0 6px 20px var(--shadow); }
.stat-label { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 750; }
.stat-value { color: var(--accent-dark); font-size: 1.3rem; font-weight: 800; margin-top: .25rem; overflow-wrap: anywhere; }
.step-card { background: var(--surface-soft); border: 1px solid var(--border); border-radius: 16px; padding: .95rem 1rem; min-height: 105px; }
.step-number { display: inline-flex; width: 30px; height: 30px; align-items: center; justify-content: center; border-radius: 50%; background: var(--accent); color: white; font-weight: 800; margin-bottom: .55rem; }
.step-title { font-weight: 800; color: var(--text); }
.step-text { color: var(--muted); font-size: .84rem; line-height: 1.45; margin-top: .2rem; }

[data-testid="stFileUploader"] { background: var(--surface); border: 2px dashed var(--border); border-radius: 20px; padding: .65rem; box-shadow: 0 8px 25px var(--shadow); }
[data-testid="stFileUploaderDropzone"] { background: var(--surface-soft); border-radius: 14px; }
[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] span { line-height: 1.35; }

.result-card { background: linear-gradient(135deg,var(--surface) 0%,var(--surface-soft) 100%); border: 1px solid var(--border); border-radius: 22px; padding: 1.35rem 1.45rem; box-shadow: 0 10px 28px var(--shadow); }
.result-label { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 800; }
.result-name { color: var(--accent-dark); font-size: 2rem; line-height: 1.15; font-weight: 850; margin: .15rem 0 .65rem; overflow-wrap: anywhere; }
.confidence { color: var(--accent); font-size: 1.8rem; font-weight: 850; }
.pill { display: inline-block; padding: .28rem .65rem; border-radius: 999px; background: var(--accent-soft); color: var(--accent-dark); font-size: .75rem; font-weight: 800; }
.info-box { background: var(--surface-soft); border: 1px solid var(--border); border-radius: 17px; padding: 1rem 1.1rem; color: var(--text); line-height: 1.55; }
.warning-box { background: var(--warning-bg); border: 1px solid var(--warning-border); border-radius: 17px; padding: 1rem 1.1rem; color: var(--warning-text); line-height: 1.55; }
.footer-note { color: var(--muted); font-size: .78rem; line-height: 1.45; text-align: center; margin-top: 2.2rem; }

[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: .55rem .75rem; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
.stButton > button { border-radius: 12px; }

@media (max-width: 760px) {
    .block-container { padding: 1rem 1rem 3rem; }
    .hero { padding: 1.6rem 1.35rem; border-radius: 22px; }
    .hero h1 { font-size: 2rem; }
    .hero p { font-size: .95rem; }
    .stat-card { min-height: 82px; }
    .stat-value { font-size: 1.08rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    classes = [
        x.strip()
        for x in (ROOT / CLASS_NAMES_PATH).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(ROOT / MODEL_PATH, map_location=device))
    model.eval()
    return model, classes, device


transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

tips = {
    "cardboard": "Keep cardboard clean and dry and place it in the appropriate recycling stream.",
    "glass": "Keep glass separate and follow local collection rules. Handle broken glass carefully.",
    "metal": "Separate metal items for recycling where collection is available.",
    "paper": "Keep paper clean and dry. Avoid mixing heavily contaminated paper with recyclable paper.",
    "plastic": "Check the plastic type and your local recycling rules. Not every plastic item is accepted everywhere.",
    "trash": "This class represents general residual waste. Follow local municipal disposal guidance.",
}


if not (ROOT / MODEL_PATH).exists():
    st.error("The trained model is not available yet.")
    st.code("python src/train.py\npython src/evaluate.py", language="powershell")
    st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ♻️ Waste AI")
    st.caption("AI-powered waste screening")
    st.markdown("---")

    st.markdown("### Model")
    st.write("**Architecture:** MobileNetV3-Large")
    st.write("**Input:** 224 × 224")
    st.write("**Classes:** 6")
    st.write("**Device:** CUDA" if torch.cuda.is_available() else "**Device:** CPU")

    st.markdown("---")
    st.markdown("### Test performance")
    st.metric("Test accuracy", "92.49%")
    st.caption("Evaluated on 253 held-out test images.")

    st.markdown("---")
    st.caption(
        "Educational prototype. Classification and disposal guidance are general "
        "information; follow local municipal recycling rules."
    )


# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
    <div class="hero-kicker">AI + Computer Vision</div>
    <h1>Smart Waste Classification</h1>
    <p>Upload a waste image and let a trained MobileNetV3-Large model identify its category and provide general disposal guidance.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Quick stats ----------
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("Model", "MobileNetV3-Large"),
    ("Input", "224 × 224"),
    ("Categories", "6 waste types"),
    ("Test accuracy", "92.49%"),
]
for col, (label, value) in zip((c1, c2, c3, c4), stats):
    with col:
        st.markdown(
            f"<div class='stat-card'><div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )


# ---------- How it works ----------
st.markdown("<div class='section-title'>How it works</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>A simple four-stage computer-vision pipeline.</div>",
    unsafe_allow_html=True,
)
steps = [
    ("1", "Upload", "Choose a clear waste image."),
    ("2", "Preprocess", "Resize and normalize the image."),
    ("3", "Classify", "MobileNetV3 predicts the waste category."),
    ("4", "Guide", "View confidence and general disposal guidance."),
]
step_cols = st.columns(4)
for col, (number, title, description) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"<div class='step-card'><div class='step-number'>{number}</div>"
            f"<div class='step-title'>{title}</div>"
            f"<div class='step-text'>{description}</div></div>",
            unsafe_allow_html=True,
        )


# ---------- Upload ----------
st.markdown("<div class='section-title'>Upload a waste image</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>For the clearest result, use a well-lit image where one main waste item is visible.</div>",
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear, well-lit image with the waste item clearly visible.",
)

if not uploaded:
    st.markdown(
        "<div class='info-box'><b>Ready to analyze.</b><br>"
        "Upload an image above to start the AI classification.</div>",
        unsafe_allow_html=True,
    )
else:
    image = Image.open(uploaded).convert("RGB")

    st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
    left, right = st.columns([1.02, 1], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded waste image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        progress = st.progress(0)
        status = st.empty()

        status.markdown("🔍 **Checking the uploaded image...**")
        progress.progress(15)
        time.sleep(0.25)

        status.markdown("🧹 **Preprocessing the waste image...**")
        progress.progress(35)
        time.sleep(0.25)

        model, classes, device = get_model()
        x = transform(image).unsqueeze(0).to(device)
        progress.progress(55)

        status.markdown("🧠 **Classifying with MobileNetV3-Large...**")
        progress.progress(70)

        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1)[0]
            values, indices = torch.topk(probs, k=min(5, len(classes)))

        progress.progress(90)
        status.markdown("📊 **Preparing your results...**")
        time.sleep(0.25)
        progress.progress(100)
        status.success("✅ **Analysis complete**")

        top_idx = int(indices[0])
        label = classes[top_idx]
        confidence = float(values[0]) * 100

        st.markdown(
            f"""
<div class="result-card">
    <div class="result-label">Predicted category</div>
    <div class="result-name">{label.title()}</div>
    <span class="pill">AI prediction</span>
    <div style="margin-top:.9rem" class="result-label">Confidence</div>
    <div class="confidence">{confidence:.2f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ---------- Top predictions ----------
    st.markdown("<div class='section-title'>Prediction breakdown</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>The model's five highest probability classes.</div>",
        unsafe_allow_html=True,
    )

    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        col1, col2 = st.columns([4.5, 1])
        with col1:
            st.write(f"**{rank}. {classes[idx].title()}**")
            st.progress(float(score))
        with col2:
            st.metric("Score", f"{score * 100:.2f}%")

    # ---------- Guidance ----------
    st.markdown("<div class='section-title'>♻️ Disposal guidance</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='info-box'><b>{label.title()}</b><br>{tips.get(label, 'Follow local waste-management guidance.')}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='warning-box'><b>Important:</b> AI classification is a screening aid. "
        "Waste rules vary by municipality, material condition, contamination, and local collection systems. "
        "Verify disposal instructions locally.</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    "<div class='footer-note'>AI-Based Waste Classification • Educational prototype • "
    "MobileNetV3-Large • 92.49% test accuracy</div>",
    unsafe_allow_html=True,
)

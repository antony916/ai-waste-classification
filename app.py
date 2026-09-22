from pathlib import Path
import time
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH
from src.model import build_model

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI Waste Classification", page_icon="♻️", layout="wide")
st.markdown("""
<style>
.stApp{background:#f7faf8}.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero{background:linear-gradient(135deg,#123b29 0%,#198754 100%);padding:2.2rem 2.4rem;border-radius:22px;color:white;margin-bottom:1.5rem}
.hero h1{margin:0;font-size:2.25rem;font-weight:750}.hero p{margin:.55rem 0 0;opacity:.9}
.card{background:white;border:1px solid #dce9e1;border-radius:18px;padding:1.25rem 1.4rem;box-shadow:0 5px 20px rgba(28,58,43,.06)}
.label{color:#6b7d73;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.prediction{color:#123b29;font-size:1.65rem;font-weight:800}.confidence{color:#198754;font-size:1.9rem;font-weight:800}
.info{background:#eef8f2;border:1px solid #cfe7d9;border-radius:16px;padding:1rem 1.1rem;color:#244b38}
.warning{background:#fff8e8;border:1px solid #f2dfad;border-radius:16px;padding:1rem 1.1rem;color:#66501f}
[data-testid="stSidebar"]{background:#edf5f0}div[data-testid="stFileUploader"]{background:white;border:2px dashed #b8d7c5;border-radius:16px;padding:.5rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    classes=[x.strip() for x in (ROOT/CLASS_NAMES_PATH).read_text(encoding="utf-8").splitlines() if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(ROOT/MODEL_PATH,map_location=device))
    model.eval()
    return model,classes,device

transform=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
tips={"cardboard":"Keep cardboard clean and dry and place it in the appropriate recycling stream.","glass":"Keep glass separate and follow local collection rules. Handle broken glass carefully.","metal":"Separate metal items for recycling where collection is available.","paper":"Keep paper clean and dry. Avoid mixing heavily contaminated paper with recyclable paper.","plastic":"Check the plastic type and your local recycling rules. Not every plastic item is accepted everywhere.","trash":"This class represents general residual waste. Follow local municipal disposal guidance."}

if not (ROOT/MODEL_PATH).exists():
    st.error("The trained model is not available yet.")
    st.code("python src/train.py\npython src/evaluate.py",language="powershell")
    st.stop()

st.markdown("<div class='hero'><h1>♻️ AI-Based Waste Classification</h1><p>Upload a waste image and let MobileNetV3-Large classify its category.</p></div>",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ♻️ Waste AI")
    st.caption("CNN-based waste screening")
    st.markdown("---")
    st.write("**Architecture:** MobileNetV3-Large")
    st.write("**Input:** 224 × 224")
    st.write("**Classes:** 6")
    st.markdown("---")
    st.caption("Educational prototype. Disposal guidance is general information; follow local municipal recycling rules.")

st.markdown("### 1. Upload waste image")
uploaded=st.file_uploader("Choose a clear waste image",type=["jpg","jpeg","png"],help="Use a well-lit image where the waste item is clearly visible.")

if not uploaded:
    st.markdown("<div class='info'><b>How it works</b><br>Upload → preprocess → CNN classification → disposal guidance.</div>",unsafe_allow_html=True)
else:
    image=Image.open(uploaded).convert("RGB")
    left,right=st.columns([1,1],gap="large")
    with left:
        st.markdown("### 2. Uploaded image")
        st.image(image,caption="Waste image",use_container_width=True)
    with right:
        st.markdown("### 3. AI analysis")
        progress=st.progress(0)
        status=st.empty()
        status.markdown("🔍 **Checking the uploaded image...**"); progress.progress(15); time.sleep(.3)
        status.markdown("🧹 **Preprocessing the waste image...**"); progress.progress(35); time.sleep(.3)
        model,classes,device=get_model()
        x=transform(image).unsqueeze(0).to(device); progress.progress(55)
        status.markdown("🧠 **Classifying the waste with MobileNetV3...**"); progress.progress(70)
        with torch.inference_mode():
            probs=torch.softmax(model(x),dim=1)[0]
            values,indices=torch.topk(probs,k=min(5,len(classes)))
        progress.progress(90); status.markdown("📊 **Preparing classification results...**"); time.sleep(.3)
        progress.progress(100); status.success("✅ **Analysis complete**")
        top_idx=int(indices[0]); label=classes[top_idx]; confidence=float(values[0])*100
        st.markdown(f"<div class='card'><div class='label'>Predicted category</div><div class='prediction'>{label.title()}</div><div class='label'>Confidence</div><div class='confidence'>{confidence:.2f}%</div></div>",unsafe_allow_html=True)
    st.markdown("### Top 5 predictions")
    for rank,(score,idx) in enumerate(zip(values.tolist(),indices.tolist()),1):
        c1,c2=st.columns([4,1])
        with c1:
            st.write(f"**{rank}. {classes[idx].title()}**"); st.progress(float(score))
        with c2:
            st.metric("Score",f"{score*100:.2f}%")
    st.markdown("### ♻️ Disposal guidance")
    st.info(tips.get(label,"Follow local waste-management guidance."))
    st.markdown("<div class='warning'><b>Important:</b> AI classification is a screening aid. Waste rules differ by municipality, material condition, contamination, and local collection systems. Verify disposal instructions locally.</div>",unsafe_allow_html=True)

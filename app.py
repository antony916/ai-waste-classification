import os,torch,streamlit as st
from PIL import Image
from torchvision import transforms
from src.model import build_model
st.set_page_config(page_title="AI Waste Classification",page_icon="♻️"); st.title("♻️ AI-Based Waste Classification System")
MODEL="artifacts/waste_mobilenetv3.pth"; CLASSES="artifacts/class_names.txt"
if not os.path.exists(MODEL): st.warning("Train first: python src/train.py"); st.stop()
names=open(CLASSES,encoding="utf-8").read().splitlines(); dev="cuda" if torch.cuda.is_available() else "cpu"; m=build_model(len(names)); m.load_state_dict(torch.load(MODEL,map_location=dev)); m.to(dev); m.eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
f=st.file_uploader("Upload waste image",type=["jpg","jpeg","png"])
if f:
 im=Image.open(f).convert("RGB"); st.image(im,use_container_width=True)
 with torch.no_grad(): p=torch.softmax(m(tf(im).unsqueeze(0).to(dev)),1)[0]
 v,i=torch.max(p,0); label=names[i.item()]; st.success(f"Predicted: {label} — {v.item()*100:.2f}%")
 tips={"cardboard":"Keep cardboard clean and dry for recycling.","glass":"Separate glass for recycling.","metal":"Separate metal for recycling.","paper":"Keep paper clean and dry for recycling.","plastic":"Check local plastic recycling rules.","trash":"Use the general waste stream unless local guidance says otherwise."}
 st.info(tips.get(label,"Follow local disposal guidance."))

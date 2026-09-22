import os,torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import *
from src.model import build_model
from src.data import WasteDataset,split_rows

def ep(model,dl,loss_fn,opt,dev,train):
    model.train(train); c=t=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev)
        if train: opt.zero_grad()
        z=model(x); loss=loss_fn(z,y)
        if train: loss.backward(); opt.step()
        c+=(z.argmax(1)==y).sum().item(); t+=len(y)
    return c/t

def main():
    os.makedirs("artifacts",exist_ok=True); ds=load_dataset(DATASET_NAME,split="train"); names=ds.features["label"].names
    open(CLASS_NAMES_PATH,"w",encoding="utf-8").write("\n".join(names)); tr,va,te=split_rows(ds); dev="cuda" if torch.cuda.is_available() else "cpu"
    m=build_model(len(names)).to(dev); tl=DataLoader(WasteDataset(tr,True),BATCH_SIZE,shuffle=True); vl=DataLoader(WasteDataset(va),BATCH_SIZE); el=DataLoader(WasteDataset(te),BATCH_SIZE)
    loss=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=LEARNING_RATE); best=0
    for e in range(EPOCHS):
        ta=ep(m,tl,loss,opt,dev,True); va=ep(m,vl,loss,opt,dev,False); print(f"Epoch {e+1}/{EPOCHS}: train={ta:.4f} val={va:.4f}")
        if va>best: best=va; torch.save(m.state_dict(),MODEL_PATH)
    print("Best validation accuracy:",best); print("Test accuracy:",ep(m,el,loss,opt,dev,False))
if __name__=="__main__": main()

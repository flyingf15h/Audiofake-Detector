import streamlit as st
import torch
from model import MultiCNN
from predict import prepInputFile

st.title("Audifake Detector")

audiofile = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audiofile:
    st.audio(audiofile, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(audiofile.read())

    model = MultiCNN()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()

    x_raw, x_fft, x_wav = prepInputFile("temp.wav")
    x_raw = torch.tensor(x_raw).unsqueeze(0).float()
    x_fft = torch.tensor(x_fft).unsqueeze(0).float()
    x_wav = torch.tensor(x_wav).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(x_raw, x_fft, x_wav)
        prob = output.item()

    st.write(f"**Prediction Score:** `{prob:.4f}`")
    if prob >= 0.5:
        st.error("FAKE")
    else:
        st.success("REAL")

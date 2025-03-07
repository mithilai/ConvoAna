import streamlit as st
import torch
import whisper
import numpy as np
import soundfile as sf
from utils import load_conversation
from model import summarize_conversation, generate_improvements

# Load Whisper model
model = whisper.load_model("medium")  # Adjust model size if needed

def audio_to_text(audio_file):
    """Convert speech from an audio file to text using Whisper"""
    # Read the audio file with soundfile (supports multiple formats)
    audio, samplerate = sf.read(audio_file, dtype="float32")  # ✅ Ensure float32 dtype

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Convert to tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32)  # ✅ Ensure correct dtype
    
    # Transcribe audio
    result = model.transcribe(audio_tensor.numpy())
    return result["text"]

st.title("📞 Customer Support Conversation Analyzer")

uploaded_file = st.file_uploader("Upload a conversation file (.txt, .mp3, .wav)", type=["txt", "mp3", "wav"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type == "text/plain":  # ✅ Correct file type check
        conversation = load_conversation(uploaded_file)
    elif file_type in ["audio/mpeg", "audio/wav"]:  # ✅ Handles both MP3 & WAV
        temp_audio_path = "temp_audio.wav" if file_type == "audio/wav" else "temp_audio.mp3"
        
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.read())  # Save the uploaded audio file
        
        conversation = audio_to_text(temp_audio_path)  # ✅ Convert audio to text
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.subheader("📜 Original Conversation")
    st.text(conversation)

    # Summarization
    if st.button("Summarize Conversation"):
        summary = summarize_conversation(conversation)
        st.subheader("📝 Summary")
        if "Error" in summary:
            st.error(summary)
        else:
            st.write(summary)

    # Improvement Suggestions
    if st.button("Generate Improvements"):
        improved_responses = generate_improvements(conversation)
        st.subheader("✨ Improved Agent Responses")
        if "Error" in improved_responses:
            st.error(improved_responses)
        else:
            for response in improved_responses:
                st.write(response)
                st.write("---")

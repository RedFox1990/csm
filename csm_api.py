from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from generator import load_csm_1b
import torchaudio
import torch
import uuid
import os

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_csm_1b(device=device)

@app.post("/generate-audio")
async def generate_audio(text: str = Form(...), speaker: int = Form(0)):
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=10_000,
    )
    filename = f"/tmp/{uuid.uuid4()}.wav"
    torchaudio.save(filename, audio.unsqueeze(0).cpu(), generator.sample_rate)
    return FileResponse(filename, media_type="audio/wav", filename="output.wav") 
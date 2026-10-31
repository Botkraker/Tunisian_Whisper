import tempfile
import wave
from pathlib import Path

import torch
from transformers import pipeline

# cuDNN's fp16 convolutions return NaN on GTX 16xx cards, which makes Whisper output only "!".
torch.backends.cudnn.enabled = False

pipe = pipeline(
    "automatic-speech-recognition",
    model="TuniSpeech-AI/whisper-tunisian-dialect",
    device=0,
    dtype=torch.float16,
)

input_path = Path(__file__).resolve().parent.parent / "sound.wav"
chunk_duration_seconds = 2

with wave.open(str(input_path), "rb") as audio:
    sample_rate = audio.getframerate()
    chunk_frames = sample_rate * chunk_duration_seconds
    settings = audio.getparams()
    transcripts = []

    while True:
        frames = audio.readframes(chunk_frames)
        if not frames:
            break

        with tempfile.NamedTemporaryFile(suffix=".wav") as chunk_file:
            with wave.open(chunk_file.name, "wb") as chunk_audio:
                chunk_audio.setparams(settings)
                chunk_audio.writeframes(frames)

            result = pipe(
                chunk_file.name,
                generate_kwargs={"num_beams": 1, "language": "ar", "task": "transcribe"},
            )
            transcripts.append(result["text"].strip())
        print(transcripts)

print(" ".join(transcripts))

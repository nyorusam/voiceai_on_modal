# run higgs audio v2 tts on modal
#
# This endpoint uses the Higgs Audio V2 generation model from Boson AI (https://github.com/boson-ai/higgs-audio).
# Higgs Audio V2 is a multimodal audio generation model that supports many capabilities including
# voice cloning, audio editing, and various audio generation tasks. This is a simple text-to-speech
# example - see the GitHub repository for more advanced usage patterns.
#
# Parameters:
# - prompt: The text to convert to speech
# - scene_description (optional): Description of the audio scene context 
#   Default: "Audio is recorded from a quiet room."
#   Example: "Audio is recorded in a bustling coffee shop with background chatter."
#
# deploy with
# modal deploy tts/higgs_endpoint.py
#
# curl -X POST --get "https://xxxxxxxx--higgs-tts-higgs-generate.modal.run" \
#   --data-urlencode "prompt=Which Workers Will AI Hurt Most: The Young or the Experienced?"


import io
from pathlib import Path

import modal

MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")

def maybe_download_higgs_model(model_storage_dir, model_id):
    """Download Higgs model if not available locally.
    (We want to avoid downloading the same model every time we start the endpoint).
    """
    from huggingface_hub import snapshot_download
    from pathlib import Path

    model_path = model_storage_dir / model_id.replace("/", "--")

    if not model_path.exists():
        print(f"Downloading model to {model_path} ...")            
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, local_dir=model_path)
        print(f"Model downloaded successfully.")
    else:
        print(f"Model already available on {model_path}.")

    return str(model_path)

# Install dependencies from boson-ai/higgs-audio repository
image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "apt-get update && apt-get install -y git",
    "git clone https://github.com/boson-ai/higgs-audio.git /tmp/higgs-audio",
    "cd /tmp/higgs-audio && pip install -r requirements.txt && pip install -e .",
    "pip install fastapi[standard] huggingface_hub"
)
app = modal.App("higgs-tts", image=image)
volume = modal.Volume.from_name("higgs-tts", create_if_missing=True)

with image.imports():
    import torch
    import torchaudio
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message
    from fastapi.responses import StreamingResponse
    from pathlib import Path

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

@app.cls(gpu="L4", scaledown_window=60 * 5, enable_memory_snapshot=True, volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=2)
class Higgs:
    @modal.enter()
    def load(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Download models if not already cached
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_higgs_model(model_dir, MODEL_PATH)
        tokenizer_path = maybe_download_higgs_model(model_dir, AUDIO_TOKENIZER_PATH)
        
        self.serve_engine = HiggsAudioServeEngine(model_path, tokenizer_path, device=device)
        print(f"Higgs models loaded from paths: {model_path}, {tokenizer_path}")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def generate(self, prompt: str, scene_description: str = None):
        # Use default scene description if none provided
        if scene_description is None:
            scene_description = "Audio is recorded from a quiet room."
        
        system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        messages = [
            Message(
                role="system",
                content=system_prompt,
            ),
            Message(
                role="user",
                content=prompt,
            ),
        ]

        # Generate audio using Higgs TTS
        output: HiggsAudioResponse = self.serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()

        # Save the generated audio to the buffer in WAV format
        torchaudio.save(buffer, torch.from_numpy(output.audio)[None, :], output.sampling_rate, format="wav")

        # Reset buffer position to the beginning for reading
        buffer.seek(0)

        # Return the audio as a streaming response with appropriate MIME type
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
        )
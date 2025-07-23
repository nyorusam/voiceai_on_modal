# Deploys a FASTAPI endpoint for Voxtral transcription on Modal GPUs.

# To deploy run: 
#  modal deploy voxtral_transcription_endpoint.py
# To test:
#  curl -X POST "https://xxxx--voxtral-asr-voxtraltranscriber-transcribe.modal.run" \
#    -F "wav=@/Users/katrintomanek/dev/audio_samples/jfk_asknot.wav" \
#    -F "language=en"

import modal
from pathlib import Path
import time
from fastapi import File, Form
import os

MODAL_APP_NAME = "voxtral-asr"

MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")
TMP_DOWNLOAD_DIR = Path("/tmp")

GPU = 'L4'
SCALEDOWN = 60 * 2 # seconds
REPO_ID = "mistralai/Voxtral-Mini-3B-2507"

def maybe_download_model(model_storage_dir, repo_id):
    """Download Voxtral model if not available locally."""
    from pathlib import Path
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    
    model_path = model_storage_dir / repo_id.replace("/", "--")
    
    if not model_path.exists():
        print(f"Downloading and saving model to {model_path} ...")
        model_path.mkdir(parents=True)
        
        # Download and save both model and processor
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(repo_id)
        
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"Model saved successfully to {model_path}")
    else:
        print(f"Model already available on {model_path}.")
    
    return str(model_path)

def transcribe_with_voxtral(processor, model, audio_file_path: str, language: str = "en"):
    """Actual transcription logic using Voxtral.
    
    Based on: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507."""
    import time
    
    t1 = time.time()
    
    print(f"Running Voxtral transcription for language: {language}")
    inputs = processor.apply_transcrition_request(language=language, audio=audio_file_path, model_id=REPO_ID)
    inputs = inputs.to(model.device, dtype=model.dtype)
    
    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    transcription = decoded_outputs[0] if decoded_outputs else ""
    
    print(f"Transcription finished in {time.time() - t1} seconds")
    print(f"Transcription: {transcription}")
    
    return {
        'result': "success",
        'transcription': transcription,
        'processing_time': time.time() - t1
    }

#############################################
# Modal service with transcription endpoints
#############################################

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "accelerate==1.9.0",
        "torch==2.7.1",
        "mistral_common==1.8.1",
        "git+https://github.com/huggingface/transformers", # need the latest version according to: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
        "librosa",
        "huggingface_hub[hf_transfer]==0.33.4",
    )
)

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)

with cuda_image.imports():
    from fastapi import File, Form
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    import torch
    import librosa
    import io
    from pathlib import Path
    import tempfile
    import os

@app.cls(
    image=cuda_image, 
    gpu=GPU, 
    scaledown_window=SCALEDOWN, 
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=10)
class VoxtralTranscriber:
    """Voxtral transcription model."""

    @modal.enter()
    def enter(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Voxtral model on {device}")
        
        # Load processor from repo_id directly (it's lightweight)
        self.processor = AutoProcessor.from_pretrained(REPO_ID)
        
        # Try to load model from cache, fallback to repo_id
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_model(model_dir, REPO_ID)
        
        try:
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                device_map=device
            )
            print(f"Voxtral model loaded from cached path: {model_path}")
        except:
            print(f"Loading from cache failed, downloading from {REPO_ID}")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                REPO_ID, 
                torch_dtype=torch.bfloat16, 
                device_map=device
            )
            # Save for next time
            self.model.save_pretrained(model_path)

    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe(self, wav: bytes=File(), language: str=Form(default="en")):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav)
            tmp_file_path = tmp_file.name
        
        try:
            result = transcribe_with_voxtral(self.processor, self.model, tmp_file_path, language)
            return result
        finally:
            os.unlink(tmp_file_path)
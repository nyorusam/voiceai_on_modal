# Deploys a FASTAPI endpoint for Nvidia NeMo ASR models on Modal GPUs.
#
# Currently, Canary and Parakeet models are supported (each with different endpoints).
# * Parakeet supports English only.
# * Canary is a multi-lingual model supporting: English, German, French, Spanish
#
# See below for how to configure which variant of either model to use.
#
# To deploy run: 
#  modal deploy nemo_endpoint.py
# To test:
#
# Canary model:
#  curl -X POST "https://xxx--nemo-asr-parakeettranscriber-transcribe.modal.run" \
#    -F "wav=@/path/to/example.wav"
#
# Parakeet model:
#  curl -X POST "https://xxx--nemo-asr-canarytranscriber-transcribe.modal.run" \
#    -F "wav=@/path/to/example.wav" \
#    -F "language=en"

import modal
from pathlib import Path
import numpy as np
from fastapi import File, Form
import os
import logging

MODAL_APP_NAME = "nemo-asr"

SAMPLE_RATE = 16000
MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")
TMP_DOWNLOAD_DIR = Path("/tmp")

GPU = 'L4'
SCALEDOWN = 60 * 2 # seconds


# This is the smallest Parakeet model; for other sizes and variants see:
# https://huggingface.co/collections/nvidia/parakeet-659711f49d1469e51546e021
PARAKEET_MODEL = 'nvidia/parakeet-tdt-0.6b-v2'

# This is the smallest Canary model; for other sizes and variants see:
# https://huggingface.co/collections/nvidia/canary-65c3b83ff19b126a3ca62926
CANARY_MODEL = 'nvidia/canary-180m-flash'



def maybe_download_model(model_storage_dir, model_name, model_type='parakeet'):
    """Download NeMo model if not available locally.
    (We want to avoid downloading the same model every time we start the endpoint).
    """
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import EncDecMultiTaskModel

    model_dir = model_storage_dir / model_name.replace("/", "_")
    model_file = model_dir / "model.bin"

    if not model_file.exists():
        print(f"Downloading model to {model_file} ...")            
        model_dir.mkdir(parents=True, exist_ok=True)
        # Download and save the model
        if model_type == 'parakeet':
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        elif model_type == 'canary':
            model = EncDecMultiTaskModel.from_pretrained(model_name=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.save_to(str(model_file))
        print(f"Model downloaded successfully.")
    else:
        print(f"Model already available at {model_file}.")

    return str(model_file)


def transcribe_with_parakeet(
    parakeet_model, 
    audio_array: np.ndarray):
    """Transcription logic for Parakeet model (English-only)."""

    import time

    t1 = time.time()
    
    print(f"Running Parakeet transcription on audio len: {len(audio_array)}")
    
    # Use NoStdStreams to suppress NeMo's verbose output
    with NoStdStreams():
        output = parakeet_model.transcribe([audio_array])
    
    transcription = output[0].text if output else ""
    
    print(f"Parakeet transcription finished in {time.time() - t1} seconds")
    print(f"Transcription: {transcription}")

    return {
        'result': "success",
        'transcription': transcription
    }


def transcribe_with_canary(
    canary_model, 
    audio_array: np.ndarray,
    language: str = "en"):
    """Transcription logic for Canary model (multilingual)."""

    import time

    t1 = time.time()
    
    print(f"Running Canary transcription for language: {language} on audio len: {len(audio_array)}")
    
    # Use NoStdStreams to suppress NeMo's verbose output
    with NoStdStreams():
        output = canary_model.transcribe([audio_array], 
                                         batch_size=1,
                                         task='asr',
                                         source_lang=language,
                                         target_lang=language, 
                                         pnc='True'  # Punctuation and Capitalization
                                         )
    
    transcription = output[0].text if output else ""
    
    print(f"Canary transcription finished in {time.time() - t1} seconds")
    print(f"Transcription: {transcription}")

    return {
        'result': "success",
        'transcription': transcription
    }


class NoStdStreams(object):
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        import sys
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush(), self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, *args):
        import sys
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


#############################################
# Modal service with transcription endpoints
#############################################

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "numpy<2",
        "librosa",
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
    )
)

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)

with image.imports():
    from fastapi import File, Form
    import librosa
    import io
    from pathlib import Path
    import numpy as np    
    import os
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import EncDecMultiTaskModel


@app.cls(
    image=image, 
    gpu=GPU, 
    scaledown_window=SCALEDOWN, 
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=10)
class ParakeetTranscriber:
    """NeMo Parakeet model for transcription."""

    @modal.enter()
    def enter(self):
        # Silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
        
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_model(model_dir, PARAKEET_MODEL)
        
        # Check if we have a saved model, otherwise load from pretrained
        if os.path.exists(model_path):
            self.nemo_model = nemo_asr.models.ASRModel.restore_from(model_path)
        else:
            self.nemo_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL)
        
        print(f"NeMo Parakeet model loaded: {PARAKEET_MODEL}")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe(self, wav: bytes=File()):
        audio_array, _ = librosa.load(io.BytesIO(wav), sr=SAMPLE_RATE)  
        return transcribe_with_parakeet(self.nemo_model, audio_array)


@app.cls(
    image=image, 
    gpu=GPU, 
    scaledown_window=SCALEDOWN, 
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=10)
class CanaryTranscriber:
    """NeMo Canary model for multilingual transcription."""

    supported_languages = ['en', 'es', 'de', 'fr']

    @modal.enter()
    def enter(self):
        # Silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
        
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_model(model_dir, CANARY_MODEL)
        
        # Check if we have a saved model, otherwise load from pretrained
        if os.path.exists(model_path):
            self.model = EncDecMultiTaskModel.restore_from(model_path)
        else:
            self.model = EncDecMultiTaskModel.from_pretrained(CANARY_MODEL)

        # Update decode params as recommended
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decode_cfg)        
        
        print(f"NeMo Canary model loaded: {CANARY_MODEL}")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe(self, wav: bytes=File(), language: str=Form(default="en")):
        audio_array, _ = librosa.load(io.BytesIO(wav), sr=SAMPLE_RATE)  
        if language not in self.supported_languages:
            return {"result": "failure - unsupported language >" + language +"<"}
        return transcribe_with_canary(self.model, audio_array, language)
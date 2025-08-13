# Deploys a FASTAPI endpoint for Whisper models (via FasterWhisper) on Modal GPUs.

# To deploy run: 
#  modal deploy whisper_endpoint.py
# To test:
#  curl -X POST "https://xxxxxx--whisper-asr-whispertranscriber-transcribe.modal.run" \
#    -F "wav=@/path/to/example.wav" \
#    -F "language=en"

import modal
from pathlib import Path
import numpy as np
from fastapi import File, Form
import os

MODAL_APP_NAME = "whisper-asr"

SAMPLE_RATE = 16000
BEAM_SIZE = 5
MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")
TMP_DOWNLOAD_DIR = Path("/tmp")


GPU = 'L4'
SCALEDOWN = 60 * 2 # seconds


def maybe_download_model(model_storage_dir, model_id):
    """Download fasterwhisper model if not available locally.
    (We want to avoid downloading the same model every time we start the endpoint).
    """
    from huggingface_hub import snapshot_download, login
    from faster_whisper.utils import download_model
    from pathlib import Path

    model_path = model_storage_dir / model_id

    if not model_path.exists():
        print(f"Downloading model to {model_path} ...")            
        model_path.mkdir(parents=True)
        download_model(model_id, output_dir=model_path)
        print(f"Model downloaded successfully.")
    else:
        print(f"Model already available on {model_path}.")

    return str(model_path)


def transcribe_with_fasterwhisper(
    fasterwhisper_model, 
    audio_array: np.ndarray,
    language, 
    get_transcript_only=False, 
    use_word_timestamps=False):
    """Actual transcription logic."""

    import numpy as np
    import time

    t1 = time.time()
    
    task = 'transcribe'
    print(f"Running transcription for language: {language} on audio len: {len(audio_array)}")
    segments, info = fasterwhisper_model.transcribe(
        audio_array,
        beam_size=BEAM_SIZE,
        language=language,
        task=task,
        condition_on_previous_text=False,
        vad_filter=True,
        word_timestamps=use_word_timestamps,
    )
    transcription = ''
    segment_texts = []
    words = []
    confidences = []
    compression_ratios = []
    
    print("Transcribed segments:")
    for segment in segments:
        transcription += segment.text + ' '
        if get_transcript_only:
            continue
        segment_texts.append(segment.text)
        confidence = np.exp(segment.avg_logprob)
        confidences.append(confidence) 
        compression_ratios.append(segment.compression_ratio)
        print(f"Next segment:compression: {segment.compression_ratio:.3f}, confidence: {confidence:.3f}, text: {segment.text}")
        if segment.words:
            for word in segment.words:
                words.append(word)
   
    transcription = transcription.strip()
    if get_transcript_only:    
        return transcription

    avg_confidence = np.mean(confidences)
    avg_compression_ratio = np.mean(compression_ratios)
    print(f"Transcription finished in {time.time() - t1} seconds")
    print(f"Transcription: {transcription}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Average compression ratio: {avg_compression_ratio:.3f}")

    return {
        'result': "success",
        'transcription': transcription,
        'segments': segment_texts,
        'confidences': confidences,
        'compression_ratios': compression_ratios,
        'words': words
        }

#############################################
# Modal service with transcription endpoints
#############################################

# We need an image with cuda 12 and cudnn 9 when using faster-whisper (ctranslatr 4.5.0 depends on it).
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]",
        "numpy",
        "librosa",
        "huggingface_hub[hf_transfer]==0.26.2",        
        "torch",
        "ctranslate2",
        "faster_whisper",
        "transformers",
    )
)

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)

with cuda_image.imports():
    from fastapi.responses import StreamingResponse
    from fastapi import File, Form
    import librosa
    import io
    from pathlib import Path
    import numpy as np    
    import os
    from faster_whisper import WhisperModel

@app.function(secrets=[modal.Secret.from_name("huggingface-secret")])
def some_function():
    os.getenv("HUGGINGFACE_TOKEN")

    
@app.cls(
    image=cuda_image, 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=GPU, 
    scaledown_window=SCALEDOWN, 
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=10)
class WhisperTranscriber:
    """Default Whisper model, can specify language or let model detect."""

    model_id = 'large-v3-turbo'

    @modal.enter()
    def enter(self):       
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_model(model_dir, self.model_id)
        self.whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")
        print(f"FasterWhisper model loaded from path: {model_path}")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe(self, wav: bytes=File(), language: str=Form(default=None), use_word_timestamps: bool=Form(default=False)):
        audio_array, _ = librosa.load(io.BytesIO(wav), sr=SAMPLE_RATE)  
        return transcribe_with_fasterwhisper(self.whisper_model, audio_array, language, get_transcript_only=False, use_word_timestamps=use_word_timestamps)      


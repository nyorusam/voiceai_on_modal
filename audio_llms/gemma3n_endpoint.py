# Runs Google's Gemma3n AudioLLM model
# https://ai.google.dev/gemma/docs/capabilities/audio#stt
#
# Deploys a FASTAPI endpoint for Gemma3n audio transcription and audio QA on Modal GPUs.
#
# Note: for this to work, you need to add a HF token into "secrets" in your Modal dashboard (Gemma requires this for downloads of the model from HuggingFace).
#
# To deploy run: 
#  modal deploy gemma3n_endpoint.py
# To test:
# (1) Transcription
# (language can be specified optionally, if done, model will translate into this language, otherwise transcribe in source language)
#  curl -X POST "https://xxxx--gemma3n-asr-gemma3ntranscriber-transcribe.modal.run" \
#    -F "wav=@/path/to/audio.wav"
#
# (2) Audio QA
# (QA is more likely to hallucinate, especially in low resource languages. Consider tweaking the prompt.)
#   curl -X POST "https://xxxx--gemma3n-asr-gemma3ntranscriber-audio-qa.modal.run" \
#     -F "wav=@/path/to/audio.wav"
#     -F "instruction=What is being said and who is speaking?"
#
# Optional parameters
# * length of generation: eg "max_generated_tokens=512" 
# * language: eg "language=English"


from fastapi import File, Form
import modal
from pathlib import Path
import os


MODAL_APP_NAME = "gemma3n-asr"

MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")
TMP_DOWNLOAD_DIR = Path("/tmp")


WARMUP_SECONDS = 30

# # E4B needs more memory, exceeds defaults of L4
# REPO_ID = "google/gemma-3n-E4B-it"
# GPU = 'A100'

# can run on L4 (no significant speedup on A100)
REPO_ID = "google/gemma-3n-E2B-it"
GPU = 'L4'

SCALEDOWN = 60 * 2 # seconds

def maybe_download_model(model_storage_dir, hf_auth_token, repo_id):
    """Download Gemma3n model if not available locally."""
    from pathlib import Path
    from huggingface_hub import snapshot_download, login
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    print("logging into HF with auth token: xxxx")
    login(token=hf_auth_token)

    model_path = model_storage_dir / repo_id.replace("/", "--")
    
    if not model_path.exists():
        print(f"Downloading and saving model to {model_path} ...")
        model_path.mkdir(parents=True)
        
        # Download and save both model and processor
        processor = AutoProcessor.from_pretrained(repo_id)
        model = AutoModelForImageTextToText.from_pretrained(repo_id)
        
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"Model saved successfully to {model_path}")
    else:
        print(f"Model already available on {model_path}.")
    
    return str(model_path)

def warmup(processor, model, seconds=1, sampling_rate=16000):
    import numpy as np
    import time

    warmup_audio = np.zeros((sampling_rate * seconds,), dtype=np.float32)  # N second of silence

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": warmup_audio},                
                {"type": "text", "text": "Wake up!"},
            ]
        }
    ]

    t1 = time.time()
    print(">> Triggered model warmup....")
    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, 
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device, dtype=model.dtype)

    _ = model.generate(**input_ids, max_new_tokens=512)
    print(f">> Warmup complete. Took {time.time()-t1} seconds.")

def transcribe_with_gemma(processor, model, audio_file_path: str, language: str = None, max_generated_tokens: int = 512):
    """Actual transcription logic using Gemma3n.
    
    Based on: https://ai.google.dev/gemma/docs/capabilities/audio#stt"""
    import time
    t1 = time.time()
    # Create language-specific prompt
    if language:
        prompt_text = f"Transcribe this audio file in {language}."
        print(f"Running Gemma3n transcription for language: {language}")
    else:
        prompt_text = "Transcribe this audio file."
        print("Running Gemma3n transcription")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_file_path},
                {"type": "text", "text": prompt_text},
            ]
        }
    ]

    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, 
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device, dtype=model.dtype)

    outputs = model.generate(**input_ids, max_new_tokens=max_generated_tokens, do_sample=False)

    result = processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Extract transcript from the model response
    transcript = result.split('model', 1)[1].strip() if result and 'model' in result else (result if result else "")
    
    print(f"Transcription finished in {time.time() - t1} seconds")
    print(f"Transcription: {transcript}")
    
    return {
        'result': "success",
        'transcription': transcript,
        'processing_time': time.time() - t1
    }

def audio_qa_with_gemma(processor, model, audio_file_path: str, instruction: str, max_generated_tokens: int = 512):
    """Audio Q&A logic using Gemma3n.
    
    Based on: https://ai.google.dev/gemma/docs/capabilities/audio#stt"""
    import time
    
    t1 = time.time()
    
    print(f"Running Gemma3n audio Q&A with instruction: {instruction}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_file_path},
                {"type": "text", "text": instruction},
            ]
        }
    ]

    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True, 
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device, dtype=model.dtype)

    outputs = model.generate(**input_ids, max_new_tokens=max_generated_tokens, do_sample=False)

    result = processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Extract answer from the model response
    answer = result.split('model', 1)[1].strip() if result and 'model' in result else (result if result else "")
    
    print(f"Audio Q&A finished in {time.time() - t1} seconds")
    print(f"Answer: {answer}")
    
    return {
        'result': "success",
        'instruction': instruction,
        'answer': answer,
        'processing_time': time.time() - t1
    }

#############################################
# Modal service with transcription endpoints
#############################################

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "build-essential", "gcc", "g++")
    .pip_install(
        "fastapi[standard]",
        "accelerate==1.9.0",
        "torch==2.7.1",
        "transformers",
        "librosa",
        "timm",
        "huggingface_hub[hf_transfer]==0.33.4",
        "pillow" #Gemma AutoImageProcessor needs PIL library even though we are only using audio here)
    )
)

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)

with cuda_image.imports():
    from fastapi import File, Form
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch
    import librosa
    import io
    from pathlib import Path
    import tempfile
    import os


@app.cls(
    image=cuda_image, 
    gpu=GPU, 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=SCALEDOWN, 
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume})
@modal.concurrent(max_inputs=10)
class Gemma3nTranscriber:
    """Gemma3n transcription model."""

    @modal.enter()
    def enter(self):
        import torch
        
        # Disable torch compilation to avoid Triton issues
        torch._dynamo.config.disable = True
        
        # Set tensor float precision for better performance
        torch.set_float32_matmul_precision('high')
        
        print(f"Loading Gemma3n model...")
        
        # Load processor from repo_id directly (it's lightweight)
        self.processor = AutoProcessor.from_pretrained(REPO_ID)
        
        # Try to load model from cache, fallback to repo_id
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_model(model_dir, os.environ["HF_TOKEN"], REPO_ID)
        
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            print(f"Gemma3n model loaded from cached path: {model_path}")
        except:
            print(f"Loading from cache failed, downloading from {REPO_ID}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                REPO_ID, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            # Save for next time
            self.model.save_pretrained(model_path)

        # trigger warmup
        warmup(self.processor, self.model, seconds=WARMUP_SECONDS)

    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe(self, wav: bytes=File(), language: str=Form(default=None), max_generated_tokens: int=Form(default=None)):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav)
            tmp_file_path = tmp_file.name
        
        result = transcribe_with_gemma(self.processor, self.model, tmp_file_path, language, max_generated_tokens)
        return result

    @modal.fastapi_endpoint(docs=True, method="POST")
    def audio_qa(self, wav: bytes=File(), instruction: str=Form(), max_generated_tokens: int=Form(default=None)):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav)
            tmp_file_path = tmp_file.name
        
        result = audio_qa_with_gemma(self.processor, self.model, tmp_file_path, instruction, max_generated_tokens)
        return result

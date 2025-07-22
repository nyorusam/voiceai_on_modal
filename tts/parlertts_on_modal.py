# modal deploy parlertts_on_modal.py

# curl -X POST --get "https://personalizedmodels--parlertts-api-example-parlertts-generate.modal.run" \
#   --data-urlencode "prompt=Which Workers Will AI Hurt Most: The Young or the Experienced?" \
#   --data-urlencode "description=friendly old woman with a deep voice" \
#   --output /tmp/n1.wav

import io
import modal

#     #"flash-attn --no-build-isolation",
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch","torchaudio","transformers", "soundfile", "parler_tts", "fastapi[standard]")

model_volume = modal.Volume.from_name("tts-model-cache", create_if_missing=True)


app = modal.App("parlertts-api-example", 
                image=image,
                volumes={"/cache": model_volume},)

with image.imports():
    import torchaudio
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf
    import time
    import os

    from fastapi.responses import StreamingResponse



# works well
MODEL_NAME = "parler-tts/parler-tts-mini-v1.1"

# # created weird sounds, no voice
# MODEL_NAME = "parler-tts/parler-tts-large-v1"


@app.cls(gpu="L4", scaledown_window=60 * 5, enable_memory_snapshot=True)
@modal.concurrent(max_inputs=2)
class ParlerTTS:
    @modal.enter()
    def load(self):

        os.environ["HF_HOME"] = "/cache"
        os.environ["TRANSFORMERS_CACHE"] = "/cache"        

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            attn_implementation="eager", # disable flash attention for now
            ).to(device="cuda")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)


    def generate_audio(self, prompt, description):
        t1 = time.time()
        input_ids = self.description_tokenizer(description, return_tensors="pt").input_ids.to(device="cuda")
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device="cuda")
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        t = time.time() - t1
        print(f"time for generation: {t} sec")
        audio_tensor = generation.cpu().squeeze()
        if audio_tensor.dim() == 1:
            # Add channel dimension if needed
            audio_tensor = audio_tensor.unsqueeze(0)  
        return audio_tensor


    @modal.fastapi_endpoint(docs=True, method="POST")
    def generate(self, prompt: str, description: str="Person with a soft female voice speaking excitedly."):


        audio_tensor = self.generate_audio(prompt, description)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, self.model.config.sampling_rate, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav"
        )


# run chatterbox tts on modal
# example from: https://modal.com/docs/examples/chatterbox_tts
#
# deploy with
# modal deploy chatterbox_tts.py
#
# use endpoint
# mkdir -p /tmp/chatterbox-tts  # create tmp directory

# curl -X POST --get "https://xxx--chatterbox-api-example-chatterbox-generate.modal.run" \
#   --data-urlencode "prompt=Which Workers Will AI Hurt Most: The Young or the Experienced?" \
#   --output /Users/katrintomanek/Downloads/sample_chatterbox.wav


import io

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "chatterbox-tts", "fastapi[standard]", "peft"
)
app = modal.App("chatterbox-api-example", image=image)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi.responses import StreamingResponse

# gpus: a10g -- works
@app.cls(gpu="a100", scaledown_window=60 * 5, enable_memory_snapshot=True)
@modal.concurrent(max_inputs=2)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(docs=True, method="POST")
    def generate(self, prompt: str):
        # Generate audio waveform from the input text
        wav = self.model.generate(prompt)

        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()

        # Save the generated audio to the buffer in WAV format
        # Uses the model's sample rate and WAV format
        ta.save(buffer, wav, self.model.sr, format="wav")

        # Reset buffer position to the beginning for reading
        buffer.seek(0)

        # Return the audio as a streaming response with appropriate MIME type.
        # This allows for browsers to playback audio directly.
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
        )


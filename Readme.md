# VoiceAI model inference

* tooling to serve voice AI models on Modal GPUs as FASTApi endpoints

## Installation

* create a new python environment: `python3.12 -m venv venv`
* `pip install -r requirements.txt`
* create [Modal](http://modal.com) account and run `modal setup` in your terminal

## Huggingface Secrets

* some models require that the endpoint is logged in to HuggingFace (from where the endpoint is retrieving the model weights).
    * when you look at the code you will see that some endpoints (like the one for Gemma3n) is already doing that
    * for this to work, you need to add a respective "secret" to your Modal workspace.
    * the secret needs to be called "huggingface-secret" and contain your huggingface API key
    * please follow [this documentation](https://modal.com/docs/guide/secrets) to add it

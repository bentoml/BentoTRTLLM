<div align="center">
    <h1 align="center">Self-host LLMs with TensorRT-LLM and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy open-source Large Language Models (LLMs) using [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), a Python API that optimizes LLM inference on NVIDIA GPUs using TensorRT engine.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model and inference logic. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites

- You have installed Python 3.10 and `pip`.
- This example uses Llama 3.1 8B Instruct. Make sure you have [gained access to the model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
- If you want to test the Service locally, you need an Nvidia GPU with at least 20G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoTRTLLM.git
cd BentoTRTLLM/llama3.1-8b-instruct
```

If you already have CUDA toolkits and Python 3.10:

```bash
# Use uv to intall dependencies
uv pip install --index-strategy unsafe-best-match -r requirements.txt

export HF_TOKEN=<your-api-key>
```

If your system doesn't have the CUDA development toolkits installed, you can do everything inside NVIDIA’s official CUDA development Docker image:

```bash
docker run --rm --ipc=host --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.4.1-devel-ubuntu22.04
```

Then, inside the container:

```bash
# Install required system packages
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

# Install Python dependencies
pip install -r requirements.txt

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve

2025-05-13T03:42:27+0000 [INFO] [cli] Loading service from default location 'service.py'
```

The server is now active at `http://localhost:3000`. You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors in plain English",
  "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information.",
  "max_tokens": 1024
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        max_tokens=1024
    )
    for response in response_generator:
        print(response, end='')
```

</details>

<details>

<summary>OpenAI-compatible endpoints</summary>

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
# client.models.list()

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain superconductors like I'm five years old"
        }
    ],
    stream=True,
)
for chunk in chat_completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

</details>

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy --secret huggingface
```

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
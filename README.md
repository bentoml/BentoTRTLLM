<div align="center">
    <h1 align="center">Self-host LLMs with TensorRT-LLM and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy open-source Large Language Models (LLMs) using [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), a Python API that optimizes LLM inference on NVIDIA GPUs.

See [here](https://github.com/bentoml/BentoML/tree/main/examples) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or LMDeploy options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites

- You have installed Python 3.10+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- You have installed Docker, which will be used to create a container environment to run TensorRT-LLM.
- If you want to test the Service locally, you need a Nvidia GPU with at least 20G VRAM.
- This example uses Llama 3. Make sure you have [gained access to the model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Set up the environment

Clone the project repo and TensorRT-LLM repo.

```bash
git clone https://github.com/bentoml/BentoTRTLLM.git
cd BentoTRTLLM/llama-3-8b-instruct
git clone -b v0.9.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
```

**Note**: To deploy Llama 3 70B AWQ, go to the [llama-3-70b-instruct](./llama-3-70b-instruct/) directory.

Create the base Docker environment to compile the model.

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
docker run --rm --runtime=nvidia --gpus all --volume ${PWD}:/TensorRT-LLM --entrypoint /bin/bash -it --workdir /TensorRT-LLM nvidia/cuda:12.1.0-devel-ubuntu22.04
```

Install dependencies inside the Docker container. Note that TensorRT-LLM requires Python 3.10.

```bash
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the stable version (corresponding to the cloned branch) of TensorRT-LLM.
pip3 install tensorrt_llm==0.9.0 -U --extra-index-url https://pypi.nvidia.com

# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
apt-get install -y git
huggingface-cli login --token *****
```

Build the Llama 8B model using a single GPU and BF16.

```bash
python3 examples/llama/convert_checkpoint.py --model_dir ./Meta-Llama-3-8B-Instruct \
            --output_dir ./tllm_checkpoint_1gpu_bf16 \
            --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/llama/8B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --max_batch_size 64 \
            --max_input_len 512 \
            --paged_kv_cache enable \
            --use_paged_context_fmha enable
```

The model should be successfully built now. Exit the Docker image.

```bash
exit
```

Clone the `tensorrtllm_backend` repo.

```bash
cd ..
git clone -b v0.9.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
```

Now, the `BentoTRTLLM/` directory should have one `TenosrRT-LLM/` directory and one `tensorrtllm_backend/` directory.

Copy the model.

```bash
cd tensorrtllm_backend
cp ../TensorRT-LLM/tmp/llama/8B/trt_engines/bf16/1-gpu/* all_models/inflight_batcher_llm/tensorrt_llm/1/
```

Set the `tokenizer_dir` and `engine_dir` paths.

```bash
HF_LLAMA_MODEL=TensorRT-LLM/Meta-Llama-3-8B-Instruct
ENGINE_PATH=tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:64

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.95,exclude_input_in_output:True,enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
```

## Import the model

Install BentoML.

```bash
pip install bentoml
```

Make sure you are in the `llama-3-8b-instruct` directory and import the model to the BentoML Model Store.

```bash
python pack_model.py
```

To verify it, run:

```bash
$ bentoml models list

Tag                                                                           Size       Creation Time
meta-llama--meta-llama-3-8b-instruct-trtllm-rtx4000:7eu4l2reqwohx3lu          45.80 GiB  2024-06-07 04:25:30
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. To serve it locally, first create a Docker container environment for TensorRT-LLM:

```bash
docker run --runtime=nvidia --gpus all -v ${PWD}:/BentoTRTLLM -v ~/bentoml:/root/bentoml -p 3000:3000 --entrypoint /bin/bash -it --workdir /BentoTRTLLM nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Start the Service.

```bash
$ bentoml serve .
2024-06-07T05:16:38+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:TRTLLM" listening on http://localhost:3000 (Press CTRL+C to quit)
I0607 05:16:39.805180 117 pinned_memory_manager.cc:275] Pinned memory pool is created at '0x7f7c64000000' with size 268435456
I0607 05:16:39.805431 117 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I0607 05:16:39.810192 117 model_lifecycle.cc:469] loading: postprocessing:1
I0607 05:16:39.810243 117 model_lifecycle.cc:469] loading: preprocessing:1
I0607 05:16:39.810385 117 model_lifecycle.cc:469] loading: tensorrt_llm:1
I0607 05:16:39.810426 117 model_lifecycle.cc:469] loading: tensorrt_llm_bls:1
I0607 05:16:39.841462 117 python_be.cc:2391] TRITONBACKEND_ModelInstanceInitialize: postprocessing_0_0 (CPU device 0)
I0607 05:16:39.841462 117 python_be.cc:2391] TRITONBACKEND_ModelInstanceInitialize: preprocessing_0_0 (CPU device 0)
[TensorRT-LLM][WARNING] gpu_device_ids is not specified, will be automatically set
[TensorRT-LLM][WARNING] max_tokens_in_paged_kv_cache is not specified, will use default value
[TensorRT-LLM][WARNING] batch_scheduler_policy parameter was not found or is invalid (must be max_utilization or guaranteed_no_evict)
[TensorRT-LLM][WARNING] enable_chunked_context is not specified, will be set to false.
...
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors like I'\''m five years old",
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

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it. Note that you need to specify the CUDA version in `bentofile.yaml`.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).

Step by step instructions to build a trtllm bento

```bash
git clone https://github.com/bentoml/BentoTRTLLM.git

######### Compile models inside BentoTRTLLM/

git clone -b v0.8.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# prepare raw models
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# Obtain and start the basic docker image environment.
docker run --rm --runtime=nvidia --gpus all --volume ${PWD}:/TensorRT-LLM --entrypoint /bin/bash -it --workdir /TensorRT-LLM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the stable version (corresponding to the cloned branch) of TensorRT-LLM.
pip3 install tensorrt_llm==0.8.0 -U --extra-index-url https://pypi.nvidia.com

# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****

# Build the Llama 8B model using a single GPU and BF16.
python3 examples/llama/convert_checkpoint.py --model_dir ./Meta-Llama-3-8B-Instruct \
            --output_dir ./tllm_checkpoint_1gpu_bf16 \
            --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/llama/8B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --max_batch_size 64 \
            --max_input_len 512 \
            --paged_kv_cache enable

####### exit previous docker image

cd ..
git clone -b v0.8.0 https://github.com/triton-inference-server/tensorrtllm_backend.git

# now BentoTRTLLM/ should have one TenosrRT-LLM/ directory and one tensorrtllm_backend/ directory

cd tensorrtllm_backend
cp ../TensorRT-LLM/tmp/llama/8B/trt_engines/bf16/1-gpu/* all_models/inflight_batcher_llm/tensorrt_llm/1/

#Set the tokenizer_dir and engine_dir paths
HF_LLAMA_MODEL=TensorRT-LLM/Meta-Llama-3-8B-Instruct
ENGINE_PATH=tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:64

python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.95,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0


########### build bento

python3 -m venv venv
source venv/bin/activate

pip install bentoml
python pack_model.py  # maybe edit file to change model tag's gpu type for clarification
bentoml build .
```

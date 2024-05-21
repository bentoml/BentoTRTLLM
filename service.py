import os
import random
import subprocess
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from pack_model import BENTO_MODEL_TAG, RAW_MODEL_DIR


MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


@bentoml.service(
    name="bentotrtllm-llama3-8b-insruct-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class TRTLLM:

    bento_model_ref = bentoml.models.get(BENTO_MODEL_TAG)

    def __init__(self) -> None:

        target_dir = self.bento_model_ref.path
        cmd = ["python3", "tensorrtllm_backend/scripts/launch_triton_server.py"]
        flags = [
            "--model_repo",
            "tensorrtllm_backend/all_models/inflight_batcher_llm",
            "--world_size",
            "1",
        ]
        self.launcher = subprocess.Popen(
            cmd + flags,
            env = {**os.environ},
            cwd=target_dir,
        )

        from transformers import AutoTokenizer

        raw_model_dir = os.path.join(
            self.bento_model_ref.path_of("TensorRT-LLM"), RAW_MODEL_DIR
        )

        tokenizer = AutoTokenizer.from_pretrained(raw_model_dir)
        self.stop_tokens = [
            tokenizer.convert_ids_to_tokens(
                tokenizer.eos_token_id,
            ),
            "<|eot_id|>",
        ]


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:

        from trtllm_client import run_inference
        import tritonclient.grpc as grpcclient
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)

        client = grpcclient.InferenceServerClient("localhost:8001")

        async for response in run_inference(
            client, prompt, output_len=max_tokens, request_id=str(random.randint(1, 9999999)),
            repetition_penalty=None, presence_penalty=None, frequency_penalty=None,
            temperature=1.0, stop_words=self.stop_tokens, bad_words=None, embedding_bias_words=None,
            embedding_bias_weights=None, model_name="ensemble", streaming=True, beam_width=1,
            overwrite_output_text=None, return_context_logits_data=None,
            return_generation_logits_data=None, end_id=None, pad_id=None, verbose=None,
        ):
            yield response

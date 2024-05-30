import json
import os
import random
import subprocess
from typing import AsyncGenerator, Optional

import bentoml
import numpy as np
import tritonclient.grpc.aio as grpcclient
from annotated_types import Ge, Le
from pack_model import BENTO_MODEL_TAG
from typing_extensions import Annotated

MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


@bentoml.service(
    name="bentotrtllm-llama3-70b-insruct-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
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
            env={**os.environ},
            cwd=target_dir,
        )
        self._grpc_client = None

    def start_grpc_stream(self) -> grpcclient.InferenceServerClient:
        if self._grpc_client:
            return self._grpc_client

        self._grpc_client = grpcclient.InferenceServerClient(
            url=f"localhost:8001", verbose=False
        )
        return self._grpc_client

    def prepare_tensor(self, name, input):
        from tritonclient.utils import np_to_triton_dtype

        t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
        t.set_data_from_numpy(input)
        return t

    def create_request(
        self,
        prompt,
        streaming,
        request_id,
        output_len,
        temperature=1.0,
    ):
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_len
        streaming_data = np.array([[streaming]], dtype=bool)
        temperature_data = np.array([[temperature]], dtype=np.float32)

        inputs = [
            self.prepare_tensor("text_input", input0_data),
            self.prepare_tensor("max_tokens", output0_len),
            self.prepare_tensor("stream", streaming_data),
            self.prepare_tensor("temperature", temperature_data),
        ]

        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("text_output"))

        # Issue the asynchronous sequence inference.
        return {
            "model_name": "ensemble",
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
        }

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)

        grpc_client_instance = self.start_grpc_stream()

        async def input_generator():
            yield self.create_request(
                prompt,
                streaming=True,
                request_id=random.randint(1, 9999999),
                output_len=max_tokens,
            )

        response_iterator = grpc_client_instance.stream_infer(
            inputs_iterator=input_generator(),
        )

        try:
            async for response in response_iterator:
                result, error = response
                if result:
                    result = result.as_numpy("text_output")
                    yield result[0].decode("utf-8")
                else:
                    yield json.dumps({"status": "error", "message": error.message()})

        except grpcclient.InferenceServerException as e:
            print(f"InferenceServerException: {e}")

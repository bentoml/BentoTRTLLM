import json
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import google.protobuf.json_format
from tritonclient.grpc.service_pb2 import ModelInferResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient


def prepare_tensor(name, input):
    from tritonclient.utils import np_to_triton_dtype

    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class StreamingResponseGenerator(queue.Queue):
    """A Generator that provides the inference results from an LLM."""

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        request_id: str,
        force_batch: bool,
        model_name: str,
        stop_words: Sequence[str],
    ) -> None:
        """Instantiate the generator class."""
        super().__init__()
        self.client = client
        self.request_id = request_id
        self._batch = force_batch
        self._stop_words = stop_words
        self._model_name = model_name

    def __iter__(self):
        """Return self as a generator."""
        return self

    def __next__(self) -> str:
        """Return the next retrieved token."""
        val = self.get()
        if val is None or val in self._stop_words:
            inputs = [
                grpcclient.InferInput("input_ids", [1, 1], "INT32"),
                grpcclient.InferInput("input_lengths", [1, 1], "INT32"),
                grpcclient.InferInput("request_output_len", [1, 1], "UINT32"),
                grpcclient.InferInput("stop", [1, 1], "BOOL"),
            ]
            inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
            inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
            inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.uint32))
            inputs[3].set_data_from_numpy(np.array([[True]], dtype="bool"))

            self.client.async_stream_infer(
                self._model_name,
                inputs,
                request_id=self.request_id,
                parameters={"Streaming": True},
            )

            self.client.stop_stream()
            raise StopIteration()
        return val


@staticmethod
def _process_result(result: Dict[str, str]) -> str:
    """Post-process the result from the server."""

    message = ModelInferResponse()
    google.protobuf.json_format.Parse(json.dumps(result), message)
    infer_result = grpcclient.InferResult(message)
    np_res = infer_result.as_numpy("text_output")

    generated_text = ""
    if np_res is not None:
        generated_text = "".join([token.decode() for token in np_res])

    return generated_text


def callback(
    result_queue: queue.Queue[Union[Optional[Dict[str, str]], str]],
    result,
    error,
    stop_words: List[str],
):
    if error:
        result_queue.put(error)
    else:
        response_raw: dict = result.get_response(as_json=True)
        # TODO: Check the response is a map rather than a string
        if "outputs" in response_raw:
            # the very last response might have no output, just the final flag
            response = _process_result(response_raw)

            if response in stop_words:
                result_queue.put(None)
            else:
                result_queue.put(response)

        if response_raw["parameters"]["triton_final_response"]["bool_param"]:
            # end of the generation
            result_queue.put(None)


async def run_inference(
    triton_client: grpcclient.InferenceServerClient,
    prompt,
    output_len,
    request_id,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    temperature,
    stop_words,
    bad_words,
    embedding_bias_words,
    embedding_bias_weights,
    model_name,
    streaming,
    beam_width,
    overwrite_output_text,
    return_context_logits_data,
    return_generation_logits_data,
    end_id,
    pad_id,
    verbose,
):

    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.int32) * output_len
    streaming_data = np.array([[streaming]], dtype=bool)
    beam_width_data = np.array([[beam_width]], dtype=np.int32)
    temperature_data = np.array([[temperature]], dtype=np.float32)

    inputs = [
        prepare_tensor("text_input", input0_data),
        prepare_tensor("max_tokens", output0_len),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
    ]

    if bad_words:
        bad_words_list = np.array([bad_words], dtype=object)
        inputs += [prepare_tensor("bad_words", bad_words_list)]

    if stop_words:
        stop_words_list = np.array([stop_words], dtype=object)
        inputs += [prepare_tensor("stop_words", stop_words_list)]

    if repetition_penalty is not None:
        repetition_penalty = [[repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32)
        inputs += [prepare_tensor("repetition_penalty", repetition_penalty_data)]

    if presence_penalty is not None:
        presence_penalty = [[presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
        inputs += [prepare_tensor("presence_penalty", presence_penalty_data)]

    if frequency_penalty is not None:
        frequency_penalty = [[frequency_penalty]]
        frequency_penalty_data = np.array(frequency_penalty, dtype=np.float32)
        inputs += [prepare_tensor("frequency_penalty", frequency_penalty_data)]

    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits", return_context_logits_data),
        ]

    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits", return_generation_logits_data),
        ]

    if (embedding_bias_words is not None and embedding_bias_weights is None) or (
        embedding_bias_words is None and embedding_bias_weights is not None
    ):
        assert 0, "Both embedding bias words and weights must be specified"

    if embedding_bias_words is not None and embedding_bias_weights is not None:
        assert len(embedding_bias_words) == len(
            embedding_bias_weights
        ), "Embedding bias weights and words must have same length"
        embedding_bias_words_data = np.array([embedding_bias_words], dtype=object)
        embedding_bias_weights_data = np.array(
            [embedding_bias_weights], dtype=np.float32
        )
        inputs.append(prepare_tensor("embedding_bias_words", embedding_bias_words_data))
        inputs.append(
            prepare_tensor("embedding_bias_weights", embedding_bias_weights_data)
        )
    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        inputs += [prepare_tensor("end_id", end_id_data)]

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        inputs += [prepare_tensor("pad_id", pad_id_data)]

    outputs = [grpcclient.InferRequestedOutput("text_output")]

    result_queue = StreamingResponseGenerator(
        triton_client,
        request_id,
        force_batch=False,
        stop_words=stop_words,
        model_name=model_name,
    )
    # Establish stream
    triton_client.start_stream(
        callback=partial(callback, result_queue, stop_words=stop_words)
    )
    # Send request
    triton_client.async_stream_infer(
        model_name, inputs=inputs, outputs=outputs, request_id=request_id
    )

    for token in result_queue:
        yield token

    # Wait for server to close the stream
    triton_client.stop_stream()

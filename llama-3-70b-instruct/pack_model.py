import os
import shutil

import bentoml

MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
RAW_MODEL_DIR = MODEL_ID.split("/")[-1]
BENTO_MODEL_TAG = MODEL_ID.lower().replace("/", "--") + "-trtllm" + "-a100"
SUBDIRS = ("TensorRT-LLM", "tensorrtllm_backend")


def ignore_patterns(path, names):
    ignored_names = []

    splited = os.path.split(path.lower())
    if splited[0] == "":
        ignored_names += [".git"]
        if path.lower() == "TensorRT-LLM".lower():
            ignored_names += ["tllm_checkpoint_1gpu_bf16"]
            ignored_names += ["tmp"]

    if splited[0] == "TensorRT-LLM".lower() and splited[1] == RAW_MODEL_DIR.lower():
        ignored_names += [name for name in names if name.endswith('.safetensors')]
        ignored_names += ["original"]

    return set(ignored_names)


def pack_model(bento_model_tag, subdirs):
    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        for subdir in subdirs:
            shutil.copytree(
                subdir, bento_model_ref.path_of(subdir), ignore=ignore_patterns
            )

if __name__ == "__main__":
    pack_model(BENTO_MODEL_TAG, SUBDIRS)

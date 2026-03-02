ARG BASE_IMAGE=nvcr.io/nvidia/vllm:25.11-py3
FROM ${BASE_IMAGE}

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    VLLM_LOGGING_LEVEL=INFO \
    VLLM_MXFP4_USE_MARLIN=1

WORKDIR /workspace

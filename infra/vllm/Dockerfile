# The official vLLM OpenAI image, pinned.
#
# v0.28.0 is the version verified against these exact weights on an
# RTX 5090. Do not drop to v0.27.1: it predates
# Qwen3_5ForConditionalGeneration and cannot load this model at all.
#
# The FlashInfer incompatibility this pin was once blamed for is real, but
# it is caused by fp8 KV cache, not by the version. See the long comment on
# --kv-cache-dtype in entrypoint.sh and docs/gpu-notes.md.
ARG VLLM_VERSION=v0.28.0
FROM vllm/vllm-openai:${VLLM_VERSION}

# curl is used by the healthcheck; the base image does not always carry it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The base image sets its own entrypoint; override it so our flag logic runs.
ENTRYPOINT ["/entrypoint.sh"]

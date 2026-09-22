#!/usr/bin/env bash
# ---------------------------------------------------------------------
# vLLM launch for Qwen3.8-27B-NVFP4 on a single RTX 5090 (32GB).
#
# READ BEFORE CHANGING ANY FLAG. Several of these are not tuning knobs;
# they are workarounds for failures that took real time to diagnose on
# this exact hardware. docs/gpu-notes.md has the full history and the
# measured numbers behind every default below.
# ---------------------------------------------------------------------
set -euo pipefail

MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-therapist-base}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

echo "── vLLM ──────────────────────────────────────────────"
echo "  model        ${MODEL_ID}"
echo "  served as    ${SERVED_MODEL_NAME}"
echo "  context      ${MAX_MODEL_LEN}"
echo "  kv dtype     ${KV_CACHE_DTYPE}"
echo "  attn backend ${ATTENTION_BACKEND}"
echo "  gpu util     ${GPU_MEMORY_UTILIZATION}"
echo "──────────────────────────────────────────────────────"

ARGS=(
  --model "${MODEL_ID}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port 8000

  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"

  # --kv-cache-dtype: DO NOT set fp8, whatever the VRAM budget in the plan
  #   says. fp8 KV makes FLASH_ATTN an invalid backend, leaving only
  #   FLASHINFER/TRITON_ATTN. vLLM then picks FlashInfer, which JIT-compiles
  #   its kernels on the FIRST REQUEST rather than at startup, needs nvcc,
  #   and takes the engine down with it. The model comes up healthy and dies
  #   on your first message — which is exactly the "starts, then first
  #   request fails" symptom in the handoff, misattributed there to the vLLM
  #   version. `auto` means no JIT path is ever reached.
  #
  #   It is also unnecessary. This is a hybrid model: only 16 of its 64
  #   layers are full attention, so bf16 KV costs 64 KiB/token. See
  #   docs/gpu-notes.md for the arithmetic.
  --kv-cache-dtype "${KV_CACHE_DTYPE}"

  # Pinned so a bad combination fails loudly at startup instead of on the
  # first inference. Confirm it took: make logs-vllm | grep -i "backend"
  --attention-backend "${ATTENTION_BACKEND}"

  # MANDATORY on this card. CUDA graph capture OOMs at startup no matter
  # what gpu-memory-utilization is set to. This is not a tuning option —
  # removing it does not trade throughput for memory, it just fails.
  --enforce-eager

  # Disables the vision tower. Qwen3.8-27B is multimodal; we are not using
  # images, and the tower is 0.86 GiB we would rather spend on KV cache.
  --language-model-only

  --enable-prefix-caching

  # Tool calling and reasoning parsers, so the agent layer can use them later.
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --reasoning-parser qwen3

  # NOTE: no --disable-log-requests. That flag was removed in vLLM 0.28;
  # request logging is off by default now and the old flag is an argparse
  # error that kills the container before the model ever loads.
)

# Multi-token prediction. Proven on this model and worth
# real latency, but it costs ~0.8 GiB of weights — off by default until the
# KV cache budget is settled. SPECULATIVE_MTP=true to turn it on.
if [ "${SPECULATIVE_MTP:-false}" = "true" ]; then
  ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

# Everything after `--` in the compose command is appended verbatim,
# which is the escape hatch for one-off experiments.
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" "$@"

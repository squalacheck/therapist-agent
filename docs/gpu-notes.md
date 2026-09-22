# GPU notes — RTX 5090 (32GB, sm_120), WSL2

Measured on this machine, not copied from the plan. Where this file and
the handoff disagree, this file is the one with arithmetic behind it.

## The weights are 22.9 GiB, not 15 GB

The handoff budgets ~15 GB for "model weights, NVFP4, vision tower
disabled". The real number, read out of the safetensors headers:

| Component | Size |
|---|---|
| MLP / experts | 8.96 GiB |
| linear-attention layers (48 of them) | 8.34 GiB |
| `lm_head` | 2.37 GiB |
| `embed_tokens` | 2.37 GiB |
| full-attention layers (16 of them) | 0.88 GiB |
| vision tower (dropped by `--language-model-only`) | 0.86 GiB |
| MTP head (only loaded with `SPECULATIVE_MTP=true`) | 0.79 GiB |
| **language model, as loaded** | **22.92 GiB** |

27B parameters at 4 bits should be ~13.5 GB, so where does the rest come
from? The NVFP4 checkpoint's `ignore` list excludes `lm_head`,
`embed_tokens` and every `linear_attn` projection from quantisation. Those
load in bf16. With a 248,320-token vocabulary, the embedding and the output
head are 2.37 GiB *each*.

Reproduce it:

```bash
python3 - <<'EOF'
import json, os, struct, glob, collections
tot = collections.Counter()
for f in sorted(glob.glob(os.path.expanduser('~/models/qwen3.8-27b-nvfp4/*.safetensors'))):
    with open(f, 'rb') as fh:
        hdr = json.loads(fh.read(struct.unpack('<Q', fh.read(8))[0]))
    for k, v in hdr.items():
        if k == '__metadata__': continue
        s, e = v['data_offsets']
        tot['vision' if 'visual' in k else 'lm'] += e - s
for g, v in tot.items(): print(f'{g:8s} {v/1024**3:6.2f} GiB')
EOF
```

## KV cache costs 64 KiB/token, and fp8 is a trap

This is a **hybrid** model. `full_attention_interval: 4` means only 16 of
its 64 layers are full attention; the other 48 are linear attention with a
constant-size SSM state that does not grow with context.

So KV is charged on 16 layers, not 64:

```
2 (K and V) x 4 kv-heads x 256 head_dim x 2 bytes  = 4096 B / token / layer
4096 x 16 layers                                   = 64 KiB / token
```

At 32K context that is 2 GiB. The handoff's "KV cache, fp8, 64K context,
~10 GB" is wrong in both directions — it overestimates the cost and then
reaches for fp8 to fix a problem that does not exist.

**Do not set `--kv-cache-dtype fp8`.** fp8 KV makes `FLASH_ATTN` an invalid
backend. vLLM falls back to FlashInfer, which JIT-compiles its kernels on
the *first request* rather than at startup, needs `nvcc`, and takes the
engine down with it. The model comes up healthy and dies on your first
message.

That is the "model starts, first request fails" symptom. It is easy to
blame on the pinned vLLM version. It is not the version; it is the KV dtype. `--kv-cache-dtype auto` plus a pinned
`--attention-backend FLASH_ATTN` means the JIT path is never reached.

## What actually fits

Total 31.84 GiB. The Windows desktop compositor holds ~2.6 GiB before
anything of ours starts, and it is not negotiable from inside WSL.

```
31.84  total
-2.6   Windows compositor
-22.9  language model weights
-2.0   activations, eager mode
-0.3   SSM state cache (48 layers x ~150 MB per sequence, 2 sequences)
─────
 ~4.0  left for KV  ->  ~64K tokens at 64 KiB/token
```

Which is why `MAX_MODEL_LEN=32768` and not 65536: 32K leaves genuine
headroom, and the ground truth is the `GPU KV cache size: N tokens` line
in the startup log, not this table. Read it after any change.

## The embedder and reranker do not fit on the card

The plan allocates 2.5 GB of VRAM to them. There is not 2.5 GB. After the
weights and the KV cache there is well under a gigabyte spare, so
`EMBED_DEVICE=cpu` is not a fallback for if they misbehave on sm_120 — it
is the only configuration that fits, and `.env` ships that way.

Batch ingestion is the case that genuinely wants the GPU, and it gets it:
`docker-compose.ingest.yml` overrides `EMBED_DEVICE=cuda`, and ingestion
runs with vLLM stopped anyway (`make down-vllm && make ingest && make up`).

## Wheels: cu128 or later, never cu124

The 5090 is sm_120. The cu124 torch wheels the scaffold originally asked
for contain no sm_120 kernels — torch imports fine and then fails on the
first `.cuda()` with *"no kernel image is available for execution on the
device"*. The ingest image builds against cu128; the rag image uses CPU
torch, which is both correct and ~3 GB smaller.

## vLLM version

Pinned to `v0.28.0`, the version verified against these weights on an
RTX 5090.

`v0.27.1` — the scaffold's original pin — predates
`Qwen3_5ForConditionalGeneration` and cannot load this model at all.

Two flag changes came with 0.28.0:

- `--disable-log-requests` was **removed**. Request logging is off by
  default now, and passing the old flag is an argparse error that kills
  the container before the model ever loads.
- `--attention-backend` is available and is pinned, so a bad combination
  fails loudly at startup instead of on the first inference.

## Flags that are not tuning options

- `--enforce-eager` — CUDA graph capture OOMs at startup on this card
  regardless of `gpu-memory-utilization`. Removing it does not trade
  throughput for memory; it just fails.
- `--language-model-only` — drops the 0.86 GiB vision tower we have no use
  for.
- `VLLM_USE_FLASHINFER_SAMPLER=0` — the sampler is a separate FlashInfer
  JIT path from attention, and it has the same failure mode.

## Sharing the GPU

The model needs nearly the whole card. Anything else holding VRAM — another
model server (`pkill -f "vllm serve"`, Ollama, LM Studio), a container
(`docker stop <name>`), a game — has to stop first.

`make up` runs `infra/scripts/gpu-guard.sh` first, which refuses to start
below 26 GB free and names what is holding the card.

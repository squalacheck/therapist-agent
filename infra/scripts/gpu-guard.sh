#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Refuse to start the stack when the GPU is already spoken for.
#
# The 27B model needs nearly the whole card. If anything else is holding
# VRAM — another model server, a game, a stray container — the load fails
# as a mid-load OOM several minutes in, which is a bad way to find out.
# Check first and name the culprit.
# ---------------------------------------------------------------------
set -euo pipefail

REQUIRED_FREE_MIB="${REQUIRED_FREE_MIB:-26000}"

c_red=$'\033[31m'; c_yellow=$'\033[33m'; c_green=$'\033[32m'; c_off=$'\033[0m'

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "${c_yellow}  nvidia-smi not found — skipping the GPU check.${c_off}"
  echo "  If you are inside WSL2, confirm the NVIDIA driver is visible with: nvidia-smi"
  exit 0
fi

read -r free total < <(
  nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
    | head -n1 | tr -d ',' | awk '{print $1, $2}'
)

printf '  GPU: %s MiB free of %s MiB\n' "$free" "$total"

if [ "$free" -ge "$REQUIRED_FREE_MIB" ]; then
  echo "${c_green}  Enough VRAM free. Starting.${c_off}"
  exit 0
fi

echo
echo "${c_red}  Not enough free VRAM.${c_off}"
echo "  Need at least ${REQUIRED_FREE_MIB} MiB, found ${free} MiB."
echo

echo "  Holding memory right now:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader | sed 's/^/    /' || echo "    (none reported)"

# Docker containers are the usual surprise — a container's VRAM does not
# always show a recognisable process name, so list them as well.
if command -v docker >/dev/null 2>&1; then
  gpu_containers=$(docker ps --filter "status=running" --format '{{.Names}}\t{{.Image}}' 2>/dev/null || true)
  if [ -n "$gpu_containers" ]; then
    echo
    echo "  Running containers (any of these may hold the GPU):"
    echo "$gpu_containers" | sed 's/^/    /'
  fi
fi

cat <<'EOF'

  Usual suspects:
    another model server (vLLM, Ollama, LM Studio)   stop it first
    a container listed above                         docker stop <name>
    games or GPU-heavy desktop apps                  close them

  Override once you know what you are doing:
    REQUIRED_FREE_MIB=20000 make up
EOF

exit 1

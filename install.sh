#!/usr/bin/env bash
# ---------------------------------------------------------------------
# One-command install.
#
#   ./install.sh            interactive
#   ./install.sh --yes      accept the defaults (builds the corpus too)
#   ./install.sh --no-corpus  skip the research corpus for now
#
# Safe to re-run: every step checks whether it is already done.
#
# What it does, in order: checks the machine, writes .env with fresh
# secrets, builds the images, downloads the model weights (~25 GB),
# optionally builds the research corpus, and starts the stack.
#
# What it never does: send anything you type into the chat anywhere.
# The network access here is for downloading software, models and
# open-access papers, and it all happens before you have said a word.
# ---------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

YES=false; CORPUS=ask
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=true ;;
    --no-corpus) CORPUS=no ;;
    -h|--help) sed -n 2,18p "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

b=$'\033[1m'; red=$'\033[31m'; grn=$'\033[32m'; yel=$'\033[33m'; off=$'\033[0m'
step() { echo; echo "${b}── $* ${off}"; }
ok()   { echo "  ${grn}ok${off}  $*"; }
warn() { echo "  ${yel}!!${off}  $*"; }
die()  { echo "  ${red}xx${off}  $*"; exit 1; }
ask()  { # ask "question" default -> echoes answer
  local q=$1 d=${2:-}
  if $YES; then echo "$d"; return; fi
  local r; read -r -p "  $q " r; echo "${r:-$d}"
}

COMPOSE=(docker compose)
COMPOSE_INGEST=(docker compose -f docker-compose.yml -f docker-compose.ingest.yml)

# ── 1. The machine ────────────────────────────────────────────────
step "Checking this machine"

command -v docker >/dev/null || die "Docker is not installed. https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || die "Docker is installed but not running (or you lack permission to use it)."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required ('docker compose', not 'docker-compose')."
ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found. An NVIDIA GPU with current drivers is required."
read -r gpu_name gpu_mib < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits \
  | head -n1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2}')
ok "GPU ${gpu_name//_/ } (${gpu_mib} MiB)"
if [ "${gpu_mib:-0}" -lt 31000 ]; then
  warn "This build is tuned for a 32 GB card (RTX 5090). With ${gpu_mib} MiB the model will not fit as configured."
  [ "$(ask 'Continue anyway? [y/N]' n)" = y ] || exit 1
fi

docker run --rm --gpus all --entrypoint true ubuntu:24.04 >/dev/null 2>&1 \
  && ok "containers can see the GPU" \
  || die "Docker cannot reach the GPU. Install the NVIDIA Container Toolkit (on Windows: Docker Desktop with WSL2 integration)."

free_gb=$(df -Pk . | awk 'NR==2 {print int($4/1048576)}')
[ "$free_gb" -ge 60 ] && ok "${free_gb} GB free disk" \
  || warn "Only ${free_gb} GB free. Models, images and the corpus need about 60 GB."

command -v python3 >/dev/null || die "python3 is required on the host (for downloading models)."
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "  installing huggingface_hub for the model download"
  python3 -m pip install --user --quiet huggingface_hub 2>/dev/null \
    || python3 -m pip install --user --quiet --break-system-packages huggingface_hub \
    || die "Could not install huggingface_hub. Try: python3 -m pip install --user huggingface_hub"
fi
ok "python3 + huggingface_hub"

# ── 2. Configuration ──────────────────────────────────────────────
step "Configuration"
if [ -f .env ]; then
  ok ".env already exists — keeping it"
else
  make --no-print-directory init
fi
set -a; . ./.env; set +a

# ── 3. Images ─────────────────────────────────────────────────────
step "Building images (the first build takes a while)"
"${COMPOSE_INGEST[@]}" build
ok "images built"

# ── 4. Models ─────────────────────────────────────────────────────
step "Downloading models"
model_dir="${MODEL_DIR:-$HOME/models}/$(basename "${MODEL_ID:-/models/qwen3.8-27b-nvfp4}")"
if ls "$model_dir"/*.safetensors >/dev/null 2>&1; then
  ok "chat model already at $model_dir"
else
  echo "  the chat model is ~23 GB — this is the long step"
  make --no-print-directory fetch-llm
fi
make --no-print-directory fetch-models

# ── 5. Research corpus ────────────────────────────────────────────
step "Research corpus"
cat <<'EOF'
  The agent grounds what it says in open-access research papers (Gottman,
  EFT, attachment, trauma, CBT/DBT). Building that library downloads them
  from PubMed Central and indexes them on your GPU: about 20–40 minutes.
  You can skip it now and run `make ingest` later; the agent works without
  it, it just cannot cite anything.
EOF
if [ "$CORPUS" = ask ]; then
  [ "$(ask 'Build the corpus now? [Y/n]' y)" = n ] && CORPUS=no || CORPUS=yes
fi
if [ "$CORPUS" = yes ]; then
  if [ -z "${CONTACT_EMAIL:-}" ]; then
    echo "  PubMed asks automated downloaders for a contact email. It is sent"
    echo "  to NCBI only, with the paper requests — never anything you say in chat."
    email=$(ask 'Contact email for PubMed:' '')
    [ -n "$email" ] || die "An email is required to build the corpus. Re-run with --no-corpus to skip."
    sed -i "s|^CONTACT_EMAIL=.*|CONTACT_EMAIL=${email}|" .env
  fi
  "${COMPOSE[@]}" stop vllm >/dev/null 2>&1 || true
  make --no-print-directory ingest
  sed -i "s|^RETRIEVAL_ENABLED=.*|RETRIEVAL_ENABLED=true|" .env
  ok "corpus built, retrieval switched on"
else
  warn "skipped. Later: make down-vllm && make ingest, then set RETRIEVAL_ENABLED=true in .env and make restart"
fi

# ── 6. Start ──────────────────────────────────────────────────────
step "Starting"
make --no-print-directory up

echo "  waiting for the chat page (the model itself takes several more minutes)"
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:3000/health >/dev/null 2>&1 && break
  sleep 5
done
make --no-print-directory prompts >/dev/null 2>&1 && ok "starter prompts installed" \
  || warn "starter prompts not installed yet — run 'make prompts' once the page loads"

cat <<EOF

${b}Installed.${off}

  1. Open ${b}http://127.0.0.1:3000${off} and create your account.
     It is local to this machine — there is no cloud account.
  2. Pick the ${b}therapist${off} model. It knows nothing about you yet and
     will start by getting to know you.
  3. Optional: ${b}make backdrop${off} for the painted background.

  The model is still loading for a few minutes after this.  make logs-vllm
  Stop everything:            make down
  Erase what it knows about you:  make forget

EOF

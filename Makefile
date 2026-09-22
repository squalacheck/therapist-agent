SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE        := docker compose
COMPOSE_INGEST := docker compose -f docker-compose.yml -f docker-compose.ingest.yml
GPU_GUARD      := ./infra/scripts/gpu-guard.sh

.PHONY: help init up down restart logs ps build fetch-models \
        fetch-llm forget \
        down-vllm up-vllm vram gpu-check model-api voices say set-voice \
        ingest ingest-dry reindex reindex-resume corpus-stats \
        today today-new backdrop backdrop-off frontend-reload \
        watch end prompts \
        shell-rag shell-ingest psql \
        test eval eval-retrieval eval-safety \
        verify verify-isolation verify-model clean nuke

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────

init: ## Create .env from the example and generate secrets
	@if [ -f .env ]; then \
		echo "  .env already exists — refusing to overwrite it."; exit 1; fi
	@cp .env.example .env
	@sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$$(openssl rand -hex 24)|" .env
	@sed -i "s|^WEBUI_SECRET_KEY=.*|WEBUI_SECRET_KEY=$$(openssl rand -hex 32)|" .env
	@sed -i "s|^HF_CACHE_DIR=.*|HF_CACHE_DIR=$$HOME/.cache/huggingface|" .env
	@tz=$$(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || echo UTC); \
		sed -i "s|^TZ=.*|TZ=$${tz:-UTC}|" .env; echo "  timezone: $${tz:-UTC}"
	@echo "  .env created with generated secrets."
	@echo "  Still to fill in by hand: CONTACT_EMAIL (required for ingestion)."

fetch-llm: ## Download the 27B chat model weights (~23 GB) into MODEL_DIR
	@python3 -c "import huggingface_hub" 2>/dev/null \
		|| { echo "  huggingface_hub not on the host python."; \
		     echo "  Install it:  python3 -m pip install --user huggingface_hub"; exit 1; }
	@set -a; [ -f .env ] && . ./.env; set +a; \
		dir="$${MODEL_DIR:-$$HOME/models}/$$(basename "$${MODEL_ID:-/models/qwen3.8-27b-nvfp4}")"; \
		repo="$${MODEL_REPO:-Inferact/Qwen3.8-27B-NVFP4}"; \
		echo "  $$repo -> $$dir"; \
		python3 -c "import sys; from huggingface_hub import snapshot_download; \
print('  ok', snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2]))" "$$repo" "$$dir"

fetch-models: ## Download the embedder + reranker into the HF cache
	@# The rag service runs on an internal network with HF_HUB_OFFLINE=1,
	@# so it can never fetch these itself. They have to be on disk first.
	@python3 -c "import huggingface_hub" 2>/dev/null \
		|| { echo "  huggingface_hub not on the host python."; \
		     echo "  Install it:  python3 -m pip install --user huggingface_hub"; exit 1; }
	@HF_HOME=$${HF_CACHE_DIR:-$$HOME/.cache/huggingface} python3 -c "\
from huggingface_hub import snapshot_download; \
[print('  ok', r, '->', snapshot_download(repo_id=r, ignore_patterns=['*.pth','*.onnx','*.gguf','*.bin'])) \
 for r in ['$${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}', '$${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}']]"
	@# BM25 too. fastembed keeps its own cache, so it needs its own step.
	@docker run --rm -v $${HF_CACHE_DIR:-$$HOME/.cache/huggingface}:/hf \
		-e FASTEMBED_CACHE_PATH=/hf/fastembed therapist-agent/ingest:latest \
		python -c "from fastembed import SparseTextEmbedding; \
SparseTextEmbedding('Qdrant/bm25'); print('  ok Qdrant/bm25')"

build: ## Build all images
	$(COMPOSE) build

# ── Running ────────────────────────────────────────────────────────

up: gpu-check ## Start the stack (checks free VRAM first)
	$(COMPOSE) up -d
	@echo
	@echo "  Model  http://127.0.0.1:8002/v1   (loading — first start takes several minutes)"
	@echo "  Agent  http://127.0.0.1:8080/v1"
	@echo "  Chat   http://127.0.0.1:3000"
	@echo
	@echo "  Watch the model come up:  make logs-vllm"

down: ## Stop the stack, keeping data
	$(COMPOSE) down

restart: down up ## Stop and start

ps: ## Show service status
	$(COMPOSE) ps

logs: ## Tail all logs
	$(COMPOSE) logs -f --tail=100

logs-vllm: ## Tail the model server
	$(COMPOSE) logs -f --tail=200 vllm

logs-rag: ## Tail the agent
	$(COMPOSE) logs -f --tail=200 rag

up-vllm: gpu-check ## Start only the model server
	$(COMPOSE) up -d vllm

down-vllm: ## Stop the model server, freeing the GPU
	$(COMPOSE) stop vllm

# ── GPU ────────────────────────────────────────────────────────────

gpu-check: ## Refuse to start if the GPU is already spoken for
	@$(GPU_GUARD)

set-voice: ## Change the Read Aloud voice: make set-voice V=bf_emma
	@# Open WebUI stores audio settings in Postgres and only reads the env
	@# vars on FIRST boot, so editing .env after that does nothing. This
	@# writes the DB directly, which is the only thing that actually works.
	@test -n "$(V)" || { echo "  usage: make set-voice V=bf_emma"; exit 1; }
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-therapist} \
		-d $${POSTGRES_DB:-therapist} -c \
		"update config set value='\"$(V)\"'::json, updated_at=extract(epoch from now()) where key='audio.tts.voice';" >/dev/null
	@$(COMPOSE) restart open-webui >/dev/null
	@sed -i "s|^TTS_VOICE=.*|TTS_VOICE=$(V)|" .env 2>/dev/null || true
	@echo "  voice set to $(V) — reload the browser tab"

voices: ## List the available Read Aloud voices
	@$(COMPOSE) exec -T tts curl -sf http://localhost:8880/v1/audio/voices \
		| python3 -c "import sys,json; v=json.load(sys.stdin); \
vs=v.get('voices') or v.get('data') or v; \
n=[x if isinstance(x,str) else (x.get('id') or x.get('name')) for x in vs]; \
print('  '+', '.join(sorted(x for x in n if x)))"
	@echo
	@echo "  Set TTS_VOICE in .env, then: make restart"
	@echo "  Prefixes: af/bf = American/British female, am/bm = male"

say: ## Speak a line in the current voice: make say T="hello there"
	@$(COMPOSE) exec -T tts curl -sf -X POST http://localhost:8880/v1/audio/speech \
		-H 'Content-Type: application/json' \
		-d "{\"model\":\"kokoro\",\"input\":\"$${T:-This is the voice the agent will read aloud in.}\",\"voice\":\"$${V:-$${TTS_VOICE:-af_heart}}\",\"response_format\":\"mp3\",\"speed\":0.95}" \
		--output /tmp/say.mp3 && $(COMPOSE) cp tts:/tmp/say.mp3 ./say.mp3 \
		&& echo "  wrote ./say.mp3"

vram: ## Show live VRAM usage per process
	@nvidia-smi --query-compute-apps=pid,process_name,used_memory \
		--format=csv,noheader || true
	@echo
	@nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
		--format=csv,noheader

# ── Corpus ─────────────────────────────────────────────────────────

ingest: ## Harvest, parse, chunk, embed and index the corpus
	@echo "  Embedding wants the whole GPU. Stop the model first if it is running:"
	@echo "     make down-vllm && make ingest && make up"
	@echo
	$(COMPOSE_INGEST) run --rm --no-deps ingest python -m ingest.cli run

ingest-dry: ## Harvest metadata only — no downloads, no indexing
	$(COMPOSE_INGEST) run --rm ingest python -m ingest.cli run --dry-run

reindex: ## Re-embed and re-index documents already on disk
	$(COMPOSE_INGEST) run --rm ingest python -m ingest.cli reindex

reindex-resume: ## Continue an interrupted index run, skipping finished documents
	@# --no-deps matters. Without it compose walks the dependency graph and
	@# starts vllm, which takes 94% of the card and wedges the embedder you
	@# just freed the GPU for.
	$(COMPOSE_INGEST) run --rm --no-deps ingest python -m ingest.cli reindex --resume

corpus-stats: ## Show what is currently in the corpus
	$(COMPOSE_INGEST) run --rm ingest python -m ingest.cli stats

# ── Daily practice ─────────────────────────────────────────────────

today: ## Show today's practice list (generates it if there isn't one)
	@curl -sf "http://127.0.0.1:8080/v1/daily" \
		| python3 -c "import sys,json; p=json.load(sys.stdin); \
print(); print('  '+p['date'], '—', p.get('intro','')); print(); \
[print(f\"  {'x' if i['status']=='done' else ' ' if i['status']=='open' else '-'}  {i['title']}\n     {i['detail']}\n\") for i in p['items']]" \
		|| echo "  no plan, and the agent could not make one — is the model up?"

today-new: ## Throw today's list away and generate a fresh one
	@curl -sf -X POST "http://127.0.0.1:8080/v1/daily/regenerate" >/dev/null \
		&& $(MAKE) --no-print-directory today

# ── Frontend ───────────────────────────────────────────────────────

backdrop: ## Set the chat backdrop to infra/webui/backdrop.webp
	@# Open WebUI stores this per user in Postgres and only ever writes it
	@# from the settings dialog, so this is the same DB poke `set-voice`
	@# does, for the same reason. Swap the image by replacing the file —
	@# the path in the setting does not change.
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-therapist} \
		-d $${POSTGRES_DB:-therapist} -c \
		"update \"user\" set settings = jsonb_set(coalesce(settings::jsonb,'{}'::jsonb), \
		 '{ui,backgroundImageUrl}', '\"/static/backdrop.webp\"'::jsonb, true)::json;" >/dev/null
	@echo "  backdrop set — reload the browser tab"

frontend-reload: ## Apply edits to infra/webui/ (css or backdrop image)
	@# A plain restart is not enough, and this is a Docker Desktop/WSL
	@# detail worth knowing: single-FILE bind mounts are staged as a copy
	@# when the container is created, so an edited custom.css keeps serving
	@# the old bytes until the container is recreated. Worse, replacing the
	@# file rather than truncating it (most editors, sed -i) swaps the inode
	@# and the mount breaks outright — the container then refuses to start
	@# with "error mounting ... no such file or directory". This fixes both.
	$(COMPOSE) up -d --force-recreate --no-deps open-webui
	@echo "  frontend reloaded — hard-reload the browser tab (ctrl+shift+r)"

backdrop-off: ## Remove the chat backdrop
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-therapist} \
		-d $${POSTGRES_DB:-therapist} -c \
		"update \"user\" set settings = (settings::jsonb #- '{ui,backgroundImageUrl}')::json;" >/dev/null
	@echo "  backdrop removed — reload the browser tab"

# ── Sessions ───────────────────────────────────────────────────────

FLAG := data/runtime/shutdown-requested

watch: ## Watch for "let's end for now" and stop the stack when it arrives
	@# The rag container cannot stop the stack itself, by design: doing so
	@# means talking to the Docker daemon, and the only way in is to mount
	@# /var/run/docker.sock — handing root-equivalent control of the machine
	@# to the one container that also holds every word of these
	@# conversations. For a stack whose backend network is internal:true
	@# precisely so content cannot leave, that is the wrong trade.
	@#
	@# So the container writes a file and this decides what to do about it.
	@rm -f $(FLAG)
	@echo "  watching for a session-end request — ctrl-c to stop"
	@echo "  (say \"let's end for now\" in the chat)"
	@while true; do \
		if [ -f $(FLAG) ]; then \
			echo "  end requested $$(cat $(FLAG))"; \
			rm -f $(FLAG); \
			$(MAKE) --no-print-directory end; \
			exit 0; \
		fi; \
		sleep 3; \
	done

end: ## Close down: stop every container, keeping all data
	@# Everything durable is already written — memory extraction runs after
	@# each turn and the close-out forces the profile — so this is only the
	@# machine going quiet, not a save.
	@rm -f $(FLAG)
	$(COMPOSE) stop
	@echo
	@echo "  stopped. the GPU is free. start again with: make up"

prompts: ## Install the chat's clickable starter prompts
	@# Open WebUI ships stock suggestions ("fun fact about the Roman
	@# Empire"), which are a strange thing to be offered by a tool you
	@# opened to talk about your relationships. Stored in Postgres, not env, so
	@# this writes the DB the way set-voice and backdrop do.
	@# Piped over stdin rather than bind-mounted: a mount would need the
	@# postgres container recreated to take effect, and restarting the
	@# database to change six strings in the UI is a poor trade.
	@$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-therapist} \
		-d $${POSTGRES_DB:-therapist} < infra/postgres/prompts/suggestions.sql >/dev/null
	@$(COMPOSE) restart open-webui >/dev/null
	@echo "  starter prompts installed — reload the browser tab"

# ── Shells ─────────────────────────────────────────────────────────

shell-rag: ## Shell into the agent container
	$(COMPOSE) exec rag /bin/bash

shell-ingest: ## Shell into the ingest container
	$(COMPOSE_INGEST) run --rm --entrypoint /bin/bash ingest

psql: ## Open a psql session
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-therapist} -d $${POSTGRES_DB:-therapist}

# ── Tests and evals ────────────────────────────────────────────────

test: ## Run the unit tests
	$(COMPOSE) exec rag pytest -q /srv/tests

eval: eval-retrieval eval-safety ## Run all evals

eval-retrieval: ## Measure recall@k, with and without reranking
	$(COMPOSE) exec rag python -m app.eval.retrieval /srv/eval/retrieval.yaml

eval-safety: ## Check crisis handling
	$(COMPOSE) exec rag python -m app.eval.safety /srv/eval/safety.yaml

# ── Verification ───────────────────────────────────────────────────

verify: verify-model verify-isolation ## Run the end-to-end checks

model-api: ## Query vLLM directly through the backend network (P=/v1/models)
	@$(COMPOSE) exec -T rag curl -sf "http://vllm:8000$${P:-/v1/models}" \
		| python3 -m json.tool

verify-model: ## Confirm the model answers
	@echo "── /v1/models (via the backend network — vllm has no host port) ──"
	@$(COMPOSE) exec -T rag curl -sf http://vllm:8000/v1/models | python3 -m json.tool
	@echo "── completion through the agent ──"
	@curl -sf http://127.0.0.1:8080/v1/chat/completions \
		-H 'Content-Type: application/json' \
		-d '{"model":"therapist","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":16,"stream":false}' \
		| python3 -m json.tool

verify-isolation: ## Prove the model and the data stores cannot reach the internet
	@# Tests vllm, which holds every full prompt, and the two data stores.
	@# It deliberately does NOT claim this of rag or open-webui: those two sit
	@# on the frontend bridge because something has to serve your browser and
	@# publish a host port, and a bridge network means NAT to the outside.
	@# Asserting otherwise would be a check that passes by not looking.
	@echo "── backend network (expect internal=true) ──"
	@docker network inspect therapist-agent_backend \
		--format '  internal={{.Internal}}'
	@echo "── egress attempts from the isolated containers (expect all to fail) ──"
	@fail=0; for svc in vllm qdrant postgres; do \
		if $(COMPOSE) exec -T $$svc timeout 8 getent hosts example.com >/dev/null 2>&1; then \
			echo "  FAIL  $$svc resolved an external host"; fail=1; \
		else \
			echo "  PASS  $$svc has no route off this machine"; \
		fi; \
	done; \
	echo "── frontend bridge (these DO have egress, by necessity) ──"; \
	echo "  note  rag, open-webui — they publish host ports; see the comment above"; \
	exit $$fail

# ── Cleanup ────────────────────────────────────────────────────────

clean: ## Remove containers and images, keeping all data
	$(COMPOSE) down --rmi local

forget: ## Erase everything the agent remembers about you, keeping the corpus
	@read -p "  This deletes all chats, memory, notes and daily lists. Type 'yes': " ok; \
		[ "$$ok" = "yes" ] || { echo "  aborted"; exit 1; }
	$(COMPOSE) down
	docker volume rm -f therapist-agent_postgres-data therapist-agent_webui-data
	@echo "  done. The next 'make up' starts from a clean slate."

nuke: ## Remove everything including the corpus and your conversations
	@read -p "  This deletes the corpus, session memory and all chats. Type 'yes': " ok; \
		[ "$$ok" = "yes" ] || { echo "  aborted"; exit 1; }
	$(COMPOSE) down -v --rmi local

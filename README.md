# therapist-agent

A private, fully local assistant for thinking through relationships,
conflict and difficult personal history. It runs a 27B open-weight model on
your own GPU, grounds what it says in open-access psychotherapy research,
remembers you across conversations, and sets a small daily practice list.

**Nothing you say to it leaves your computer.** There is no cloud service,
no account with anyone, and no telemetry. The model and the databases that
hold your conversations sit on a Docker network with no route to the
internet — `make verify-isolation` proves it on your machine.

> **This is not therapy, and it is not a crisis service.** It is an AI tool.
> It cannot diagnose, it can be wrong, and it has no one checking its work.
> If you are in danger or thinking about ending your life, contact local
> emergency services or a crisis line now — in the US, call or text **988**;
> elsewhere, <https://findahelpline.com> lists services by country.

## What you need

- An **NVIDIA GPU with 32 GB of VRAM** (built and tested on an RTX 5090).
  Smaller cards will not fit the model as configured.
- Linux, or Windows with **WSL2** and **Docker Desktop** (WSL integration on).
- Docker with Compose v2 and the NVIDIA Container Toolkit.
- About **60 GB** of free disk, and a good connection for the first install.

## Install

```bash
git clone <this repo> therapist-agent
cd therapist-agent
./install.sh
```

The installer checks your machine, generates fresh secrets, builds the
images, downloads the model (~23 GB), offers to build the research library
(~20–40 minutes), and starts everything. It is safe to re-run.

Then open <http://127.0.0.1:3000>, create your account (it exists only on
your machine), and pick **therapist**.

## A clean slate

A new install knows nothing about you. The first conversation starts the way
a first session does: it introduces itself, asks what you would like to be
called, and gets to know you one question at a time. What you tell it is
kept as running notes in the local database, so the next conversation picks
up where you left off.

| Model in the picker | What it does |
|---|---|
| `therapist` | The main one. Remembers you across conversations. |
| `therapist-daily` | Opens on today's practice list, then talks about it. |
| `therapist-fresh` | Reads nothing and writes nothing — a conversation that is never remembered. |

## Your data

Everything lives in local Docker volumes on your machine:

- `make forget` — erase everything it knows about you (chats, notes, daily
  lists), keeping the research library. The next start is a first meeting.
- `make nuke` — remove everything, including the library and images.
- Say **"let's end for now"** in a chat to save and close the session.
  With `make watch` running, that also stops the stack and frees the GPU.

Network access happens only during install and `make ingest`: downloading
software, model weights and open-access papers. When building the library,
PubMed is sent the contact email you give the installer, as NCBI requires —
never anything from your conversations.

## Everyday commands

```
make up / make down     start and stop
make logs-vllm          watch the model load (a few minutes from cold)
make today              today's practice list
make set-voice V=bf_emma   change the Read Aloud voice (make voices lists them)
make backdrop           the painted chat background
make help               everything else
```

## How it works

```
browser ─► Open WebUI ─► rag (agent) ─┬─► vLLM  (27B model, GPU)
             :3000         :8080      ├─► Qdrant (research library)
                                      ├─► Postgres (memory, notes, daily list)
                                      └─► Kokoro (voice)
          └───────── frontend ────────┘└──── backend: internal, no internet ────┘
```

Each turn: a crisis screen runs first on what you said; research passages
and what it remembers about you are fetched in parallel; the reply streams
back and is screened again on the way out; then durable facts are extracted
and the running notes updated, after you have your answer.

The stance it takes — direct rather than agreeable, curious about the other
person's side, no diagnosing, prose not bullet points — is in
`services/rag/app/prompts/`. Every prompt is a plain file, and
`docs/REBUILD_PROMPT.md` is the full design specification with the reasons
behind each decision. `docs/gpu-notes.md` explains the GPU settings; several
of them look like tuning and are not.

## Known limits

- The crisis resources shown on a safety turn are US services (988, the
  National Domestic Violence Hotline). If you are elsewhere, edit
  `RESOURCES` in `services/rag/app/safety.py` for your country.
- One person per install. Memory is not separated between accounts, so do
  not share an install.
- The research library is what open-access literature covers; it cites what
  it retrieves, and says so when it is answering from general knowledge.

## For contributors

Run `scripts/install-hooks.sh` after cloning. It installs a check that
blocks any commit containing personal data — `.env`, data files, home paths,
email addresses, and any terms in your own denylist at
`~/.config/therapist-agent-private/denylist.txt` (kept outside the repo).

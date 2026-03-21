# Local LLM Orchestration Pipeline

A fully local, privacy-preserving LLM orchestration pipeline for software development
and ideation. Runs five open-weight models on consumer hardware (RTX 3080, 10 GB VRAM).

## Architecture

```
Input → [Vision Decode] → [Classify] → [Ideation*] → [Plan]
      → [Draft] → [Appraise] → [BugFix]
      → [Critique Ensemble*] → [Synthesise*] → [Validate] → Output

* = optional / mode-dependent
```

**Models:**

| Role | Model | Quant | Source |
|------|-------|-------|--------|
| Vision + Plan + Validate | Qwen3.5-9B | Q6_K | bartowski |
| Ideation (long mode) | Qwen3.5-27B | IQ2_S | bartowski |
| Draft generation | Qwen3.5-35B MoE | UD-Q4_K_XL | unsloth |
| Correctness appraisal | DeepCoder 14B | Q4_K_M | bartowski |
| Bug fix + craft review | Qwen2.5 Coder 14B | Q4_K_M | unsloth |

## Requirements

- WSL2 on Windows (Ubuntu 22.04+)
- NVIDIA RTX 3080 (10 GB VRAM) or equivalent
- 64 GB system RAM
- Python 3.11 (via conda)
- llama.cpp (built with CUDA support)
- Docker (for Langfuse)

## Setup

### 1. WSL2 configuration

Add to `%USERPROFILE%\.wslconfig` on Windows:

```ini
[wsl2]
memory=56GB
processors=12
```

### 2. Conda environment

```bash
conda create -n pipeline python=3.11
conda activate pipeline
pip install uv
uv pip install -r requirements.txt
```

### 3. Model files

Download model GGUFs to `~/models/`:

```
~/models/
  qwen3.5-9b-q6_k.gguf
  qwen3.5-27b-iq2_s.gguf
  qwen3.5-35b-moe-ud-q4_k_xl.gguf
  deepcoder-14b-q4_k_m.gguf
  qwen2.5-coder-14b-instruct-q4_k_m.gguf
```

Set `MODEL_DIR` if your models are elsewhere:

```bash
export MODEL_DIR=/path/to/models
```

### 4. Langfuse (observability)

```bash
# In WSL2
docker run -d \
  -p 3000:3000 \
  -e NEXTAUTH_SECRET=your-secret \
  -e SALT=your-salt \
  --name langfuse \
  langfuse/langfuse:latest
```

Open http://localhost:3000 and create a project. Copy the keys to `.env`.

### 5. Environment

```bash
cp .env.example .env
# Edit .env with your Langfuse keys
```

### 6. Start model servers

```bash
bash servers/start_all.sh
```

This starts all five llama.cpp server instances and health-checks each one.
Expect 2-5 minutes for the 35B to load.

### 7. Initialise database

```bash
python -c "from storage.db import initialise; initialise()"
```

## Usage

```bash
# Simple coding task (auto-classified as short mode)
python run.py "write a Python function to parse a config file"

# Force long mode for complex architectural work
python run.py --mode long "design a caching layer with Redis and fallback to in-memory"

# With an image input
python run.py --image ./mockup.png "implement this UI component"

# Skip ensemble for speed
python run.py --no-ensemble "fix the off-by-one error in my binary search"

# Read task from file
cat task.txt | python run.py --stdin

# Write output to file
python run.py --output result.py "write a CSV parser"

# Verbose debug logging
python run.py --log-level DEBUG "your task"
```

## Project Structure

```
pipeline/
├── config/
│   ├── models.yaml          # All model configuration
│   ├── routing.yaml         # Routing rules and thresholds
│   ├── llama_flags/         # llama.cpp server launch scripts
│   └── prompts/             # Prompt templates (one per role)
├── schemas/                 # Pydantic contracts for all stage I/O
├── nodes/                   # One file per pipeline stage
├── pipeline/
│   ├── state.py             # PipelineState TypedDict
│   ├── graph.py             # LangGraph graph definition
│   ├── guards.py            # Deterministic checks
│   └── routers.py           # Conditional edge functions
├── clients/
│   └── llm.py               # Central model call wrapper
├── storage/
│   ├── schema.sql           # SQLite table definitions
│   ├── db.py                # Connection management
│   ├── critique_store.py    # CritiqueRecord persistence
│   └── lesson_store.py      # LessonL memory system
├── servers/
│   ├── start_all.sh         # Start all model servers
│   └── stop_all.sh          # Stop all model servers
├── tests/
│   └── unit/                # Schema, guard, router tests
├── tools/
│   └── derive_nowait_tokens.py  # NoWait token ID derivation
├── runs/                    # Pipeline run artefacts
├── run.py                   # CLI entry point
└── requirements.txt
```

## Run Artefacts

Each run creates `runs/{uuid}/`:

```
run.json               # Run metadata and status
planspec.json          # Implementation plan (persists entire run)
draft.json             # 35B draft (overwritten on redraft)
appraisal_report.json  # DeepCoder correctness appraisal
fixed.json             # Coder 14B bug-fixed output
critique.json          # Full CritiqueRecord with all verdicts
verdict.json           # Final ValidationVerdict
stages.log             # Per-stage timing and token counts
*_thinking.log         # Thinking block output per stage
```

## Tuning

### PlanSpec depth

The most important prompt to tune is `config/prompts/plan.yaml`.
Start sparse and add fields where you observe the 35B making
unintended decisions. Edit the YAML — no code changes needed.

### NoWait tokens

After the 35B is running:

```bash
python tools/derive_nowait_tokens.py --model 35b --samples 32
```

Paste the output into `config/models.yaml` under `models.35b.nowait_tokens`.

### Lesson retrieval weights

After 50+ runs, enable Phase 2 in `config/routing.yaml`:

```yaml
lesson_retrieval:
  inject_enabled: true
```

Tune scoring weights based on which retrieved lessons the 9B
actually uses (visible in `plan_thinking.log`).

### Thread count for 35B

The `-t` flag in `config/llama_flags/35b.sh` controls CPU threads
for expert computation. Start at `physical_cores / 1.5` and sweep.

## Running Tests

```bash
pytest tests/unit/ -v
```

No model servers required for unit tests.

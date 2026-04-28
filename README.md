# Eolophus — Local LLM Orchestration Pipeline

A fully local, privacy-preserving LLM orchestration pipeline for software development,
ideation, and chess analysis. Runs five open-weight models on consumer hardware
(RTX 3080, 10 GB VRAM, 64 GB RAM).

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Models](#models)
3. [Pipeline Flows](#pipeline-flows)
4. [Inline Prefix Syntax](#inline-prefix-syntax)
5. [CLI Reference](#cli-reference)
6. [Chess Mode](#chess-mode)
7. [HTTP Server](#http-server)
8. [Configuration](#configuration)
9. [Setup](#setup)
10. [Performance Notes](#performance-notes)
11. [Run Artefacts](#run-artefacts)
12. [Running Tests](#running-tests)

---

## How It Works

Every run passes through a LangGraph directed graph. Nodes do the work; routers
decide what happens next. Models load on demand and swap out when a different
model is needed — only one lives in VRAM at a time on a 10 GB card.

```
Input
  └─ [Vision Decode?] ──────── if image attached
  └─ [Classify]       ──────── 9B determines mode, type, complexity
       ├─ [Ideation]  ──────── 27B — long/ideation or long/mixed only
       └─ [Plan]      ──────── 9B thinking — consistency check + PlanSpec
            ├─ [Draft]         ──────── 35B MoE — long mode
            ├─ [Draft Short]   ──────── 9B — short mode
            └─ [Sub-spec runner] ─────── decomposed tasks only
                 └─ [Final Validate]
  └─ [Appraise]     ──────── DeepCoder 14B — coding tasks only
  └─ [BugFix]       ──────── Coder 14B — coding tasks only
  └─ [Critique A?]  ──────── 9B coherence — complex long mode
  └─ [Critique B?]  ──────── DeepCoder correctness — complex long mode
  └─ [Synthesise?]  ──────── 35B or 9B — if ensemble ran
  └─ [Validate]     ──────── 9B — routing decision
       ├─ pass        → Output
       ├─ minor_fix   → BugFix (short) or Draft (long, if appraisal was severe)
       ├─ spec_problem → Plan
       └─ unresolvable → Output + failure flag
```

**Key behaviours:**
- The 9B is the workhorse — it classifies, plans, validates, and runs both critic roles
- The 35B only runs in long mode and only for drafting (and complex synthesis)
- DeepCoder and Coder 14B are skipped for non-coding task types
- Correction loops route back to Coder 14B (minor issues) or 35B (severe issues)
- Thinking budgets are complexity-aware — simple tasks get less thinking time
- LLMLingua-2 compresses ideation output and correction history before they enter prompts
- LessonL stores lessons after each run; injects top-k into planning after 50+ runs

---

## Models

| Role | Model | Quant | Port |
|------|-------|-------|------|
| Classify / Plan / Validate / Critics | Qwen3.5-9B | Q6_K (bartowski) | 8081 |
| Ideation | Qwen3.5-27B | IQ2_S (bartowski) | 8082 |
| Draft / Complex Synthesis | Qwen3.5-35B MoE | UD-Q4_K_XL (unsloth) | 8083 |
| Appraise / Critic B | DeepCoder 14B | Q4_K_M (bartowski) | 8084 |
| BugFix | Qwen2.5 Coder 14B | Q4_K_M (unsloth) | 8085 |

All five roles and their model assignments are configurable in `config/models.yaml`.
Change a model assignment without touching code.

---

## Pipeline Flows

### Short Mode
Fastest path. The 9B handles everything — no 35B, no DeepCoder, no Coder 14B.
Suitable for simple, well-defined tasks.

```
Classify → Plan → Draft (9B) → Validate → Output
```

Correction loop routes back to Coder 14B (via 9B proxy) for minor fixes.

### Long / Coding
Full execution pipeline. The 35B drafts, DeepCoder appraises, Coder 14B fixes.

```
Classify → Plan → Draft (35B) → Appraise → BugFix → [Ensemble?] → Validate → Output
```

Correction loop on minor_fix: routes to BugFix if appraisal was clean,
routes back to Draft (35B) only if appraisal found critical or major issues.

### Long / Ideation
Adds the 27B exploration stage before planning.

```
Classify → Ideation (27B) → Plan → Draft (35B) → Appraise → BugFix → Validate → Output
```

The 27B explores the problem space broadly. The 9B consistency-checks ideation output
before planning — infeasible or contradictory ideas are dropped here.

### Long / Mixed
Same as ideation — 27B fires because the task has both open-ended and implementation
components. The planning prompt signals this to treat breadth and implementation
with equal weight.

```
Classify → Ideation (27B) → Plan → Draft (35B) → Appraise → BugFix → Validate → Output
```

### Long with Critique Ensemble
Triggered automatically for complex tasks in long mode (configurable in routing.yaml).
Two independent critics evaluate the fixed output before the validator decides.

```
... → BugFix → Critic A (9B coherence) → Critic B (DeepCoder correctness)
             → Synthesise → Validate → Output
```

Critic A and B receive FixedOutput independently — no cross-visibility before synthesis.
The synthesis model consolidates verdicts, escalating to the most serious one.

### Decomposed Tasks
When the 9B determines a task has more than 5 independent components, it sets
`decompose=true`. Each component runs as a separate short-mode sub-spec pipeline.
Final validation assembles all outputs and checks interface compatibility.

```
Classify → Plan → Sub-spec Runner
  ├─ Component A: Plan → Draft → Appraise → BugFix → Validate
  ├─ Component B: Plan → Draft → Appraise → BugFix → Validate
  └─ ...
→ Final Validate (interface compatibility check + coherence) → Output
```

### Chess Mode (Fast)
Direct path — bypasses the pipeline graph entirely. Hits the 9B with a chess-specific
prompt. Designed for per-move latency. No model swaps.

```
ChessCoachingRequest → 9B (thinking, no NoWait) → ChessAnalysisOutput
```

### Chess Mode (Slow)
For flagged moves (blunders >150cp, depth mirages, sacrifices). Larger thinking
budget, deeper analysis. Still direct — no planning or drafting stages.

```
ChessCoachingRequest → 9B (full thinking budget, no NoWait) → ChessAnalysisOutput
```

---

## Inline Prefix Syntax

Prefixes are parsed from the task string itself. No flags needed.

```bash
# Force mode
python run.py "/long design a caching layer with Redis and memory fallback"
python run.py "/short fix the off-by-one in binary_search()"

# Force mode + task type
python run.py "/long/coding implement a rate limiter with token bucket algorithm"
python run.py "/long/ideation explore approaches for a distributed task queue"
python run.py "/long/mixed design and implement a simple event bus in Python"
python run.py "/short/coding write a CSV parser with error handling"

# Skip ensemble
python run.py "/no-ensemble fix the null check in process()"

# Combine
python run.py "/no-ensemble/long implement a simple LRU cache"
```

**Priority order:** CLI flags > inline prefix > auto-classification by 9B

The 9B still classifies even when mode/type are pinned — it determines complexity
and decompose, which the prefix cannot override.

---

## CLI Reference

```bash
python run.py [OPTIONS] [TASK]
```

| Option | Description |
|--------|-------------|
| `TASK` | Task string. Omit to read from stdin. |
| `--mode short\|long` | Force pipeline mode |
| `--task-type coding\|ideation\|mixed` | Force task type |
| `--image PATH` | Attach an image for vision decode |
| `--no-ensemble` | Skip critique ensemble |
| `--stdin` | Read task from stdin |
| `--output PATH` | Write output to file instead of stdout |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Log verbosity (default: INFO) |

**Examples:**

```bash
# From file
cat task.txt | python run.py --stdin --output result.py

# With image
python run.py --image ./mockup.png "/long/coding implement this UI component"

# Debug a specific run
python run.py --log-level DEBUG "/short write a fibonacci function"
```

---

## Chess Mode

Chess analysis is exposed via the HTTP server, not the CLI. The pipeline runs on the
same machine as Dacelo's backend (WSL2), and Dacelo calls the REST endpoint directly.

### What changes vs the Swift implementation

| | Swift LLMHookService | Eolophus chess node |
|---|---|---|
| Board state | tacticalFlags only (pre-digested) | + coordinate list (White: King e1, ...) |
| Reasoning | None — direct JSON output | `internal_reasoning` field forces CoT first |
| NoWait suppression | Applied | **Disabled** — chess needs backtracking |
| Thinking mode | Configurable | Always on (budget differs by mode) |

### Board state format

Pass the `pieces` field in your request alongside the existing `tacticalFlags`:

```json
{
  "pieces": {
    "white": ["King e1", "Queen d1", "Rook h1", "Knight f3", "Pawns e4 d4 c3"],
    "black": ["King e8", "Queen d8", "Rook a8", "Bishop c5", "Pawns e5 d6 c7"]
  }
}
```

The model receives this as a coordinate list — more token-efficient than ASCII boards
and unambiguous about piece relationships.

### Response format

The pipeline returns the standard `ChessCoachingOutput` fields. The `internal_reasoning`
field is generated but not returned to the Swift caller — it exists purely to force
chain-of-thought before the headline/explanation fields are written.

### Fast vs Slow mode

Controlled by `isSlowMode` in the request:
- `false` — short thinking budget, concise output, fires every move
- `true`  — full thinking budget, detailed output, fires for flagged moves only

---

## HTTP Server

A FastAPI server wraps both the chess endpoint and the full pipeline.

```bash
# Start the server
uvicorn server:app --host 0.0.0.0 --port 9000 --workers 1
```

**Important:** `--workers 1` is required. The model manager is a single-GPU singleton —
multiple workers would fight over VRAM.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chess/analyse` | Chess move analysis (fast or slow) |
| `POST` | `/pipeline/run` | Start a full pipeline run (async) |
| `GET` | `/pipeline/status/{run_uuid}` | Check run status and retrieve output |
| `GET` | `/health` | Liveness check |

### Chess endpoint

```bash
curl -X POST http://localhost:9000/chess/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "movePlayed": "e2e4",
    "side": "white",
    "moveNotation": "1.",
    "classification": "Good",
    "evalAfter": 0.15,
    "tacticalFlags": ["Central pawn advance"],
    "isSlowMode": false
  }'
```

### Pipeline endpoint

```bash
# Start run (returns immediately with run_uuid)
curl -X POST http://localhost:9000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"task": "/long/coding implement a rate limiter class"}'

# Poll status
curl http://localhost:9000/pipeline/status/{run_uuid}
```

---

## Configuration

### `config/models.yaml`

Single source of truth for all model configuration — ports, quantisation, thinking
defaults, NoWait token IDs. Change a model or port here; no code changes needed.

The `roles` section maps pipeline roles to model IDs. To point a role at a different
model, change one line:

```yaml
roles:
  bugfix: "9b"   # route bug fixes to 9B instead of Coder 14B
```

### `config/routing.yaml`

All routing rules, thresholds, and tunable parameters.

**Thinking budgets** — complexity-aware, per stage:
```yaml
thinking_budgets:
  plan:
    simple:   512
    moderate: 1024
    complex:  2048
  draft:
    simple:   1024
    moderate: 2048
    complex:  4096
```
Lower values = faster but shallower reasoning. Set to 0 to disable thinking for a stage.

**Correction loop** — controls when minor_fix routes back to 35B vs Coder 14B:
```yaml
correction_loop:
  minor_fix_35b_severity_threshold: 0
  # If appraisal critical+major count <= threshold, route minor_fix to bugfix
  # instead of re-drafting with 35B. Prevents unnecessary 35B reloads.
```

**Ensemble triggers** — when the critique ensemble fires:
```yaml
ensemble:
  trigger_on_complexity: ["complex"]
  trigger_on_mode: ["long"]
  force_on_final_iteration: true
```

**HTTP timeout** — must exceed the longest possible generation:
```yaml
http:
  timeout_seconds: 7200   # 2 hours — 35B draft can take 90+ min
```

**Lesson retrieval** — LessonL Phase 2 (enable after 50+ runs):
```yaml
lesson_retrieval:
  inject_enabled: true   # false by default until lessons accumulate
```

---

## Setup

### 1. Prerequisites

- Windows 11 with WSL2 (Ubuntu 22.04+)
- NVIDIA GPU driver 525.60+ on Windows
- CUDA 13.x toolkit in WSL2

### 2. WSL2 memory config

In `%USERPROFILE%\.wslconfig` on Windows:

```ini
[wsl2]
memory=56GB
processors=12
swap=8GB
```

### 3. llama.cpp

```bash
cd ~/local-llama
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DGGML_NATIVE=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON \
  -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=ON \
  -DLLAMA_CURL=ON -DGGML_LTO=ON -G Ninja

cmake --build build --config Release -j $(nproc)

# Add to PATH
sudo ln -sf ~/local-llama/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Fix shared library
echo "$HOME/local-llama/llama.cpp/build/bin" | sudo tee /etc/ld.so.conf.d/llama.conf
sudo ldconfig
```

### 4. Python environment

```bash
conda create -n llama python=3.11
conda activate llama
pip install uv
cd ~/local-llama/Eolophus
uv pip install -r requirements.txt
```

### 5. Model files

Download GGUFs to `~/local-llama/llama.cpp/models/`:

| Filename | Source |
|----------|--------|
| `qwen3.5-9b-q6_k.gguf` | bartowski / HuggingFace |
| `qwen3.5-27b-iq2_s.gguf` | bartowski / HuggingFace |
| `qwen3.5-35b-moe-ud-q4_k_xl.gguf` | unsloth / HuggingFace |
| `deepcoder-14b-q4_k_m.gguf` | bartowski / HuggingFace |
| `qwen2.5-coder-14b-instruct-q4_k_m.gguf` | unsloth / HuggingFace |

Update `~/.bashrc`:
```bash
export MODEL_DIR=$HOME/local-llama/llama.cpp/models
conda activate llama
```

### 6. Environment file

```bash
cp .env.example .env
# Edit .env — add Langfuse keys if using observability
```

Minimum `.env`:
```bash
MODEL_DIR=/home/yourname/local-llama/llama.cpp/models
LANGFUSE_HOST=http://localhost:3000
LOG_LEVEL=INFO
```

### 7. Initialise database

```bash
python -c "from storage.db import initialise; initialise()"
# Creates ~/.pipeline/pipeline.db
```

### 8. First run

The model manager starts servers on demand — no need to pre-launch anything.

```bash
python run.py "/short write a Python function that returns the fibonacci sequence"
```

Watch `runs/{uuid}/stages.log` in a second terminal to follow progress:

```bash
tail -f runs/$(ls -t runs/ | head -1)/stages.log
```

---

## Performance Notes

### Model load times (RTX 3080)

| Model | Load time |
|-------|-----------|
| 9B Q6_K | ~12s |
| 27B IQ2_S | ~12s |
| 35B MoE UD-Q4_K_XL | ~180s |
| DeepCoder 14B Q4_K_M | ~12s |
| Coder 14B Q4_K_M | ~12s |

The 35B is slow to load because its ~20GB expert weights must be staged into RAM.
Keeping it loaded between runs (when doing multiple long tasks) avoids repeated
180-second load penalties.

### Typical run times

| Flow | Approximate time |
|------|-----------------|
| short — simple task | 2-5 min |
| long/coding — moderate, no corrections | 30-45 min |
| long/mixed — with ideation | 45-90 min |
| long/coding — with correction loop | Add 30-45 min per iteration |
| chess fast | 30-60s (model already warm) |
| chess slow | 2-4 min |

### Reducing correction loop cost

The most expensive scenario is a `minor_fix` verdict routing back to the 35B.
Controlled by `minor_fix_35b_severity_threshold` in `routing.yaml`:

```yaml
correction_loop:
  minor_fix_35b_severity_threshold: 0
  # 0 = only route to 35B if appraisal found critical or major issues
  # 1 = route to 35B if appraisal found any issues including minor
```

With threshold=0, a `minor_fix` verdict on a draft that DeepCoder rated as high
satisfaction routes to Coder 14B for an in-place fix — saving a 180s 35B reload.

### Model file location matters

Models on `/mnt/c/` or `/mnt/d/` (Windows NTFS) load significantly slower than
models on WSL2's native ext4 filesystem (`~/`). If load times are slow, move model
files to the Linux filesystem.

---

## Run Artefacts

Each run creates `runs/{uuid}/`:

```
run.json                # Metadata: mode, status, timings, prefix overrides
planspec.json           # 9B's implementation plan — persists entire run
ideation.json           # 27B output — long/ideation and long/mixed only
draft.json              # 35B draft — overwritten on each correction iteration
appraisal_report.json   # DeepCoder appraisal — coding tasks only
fixed.json              # Coder 14B output — coding tasks only
critique.json           # Full CritiqueRecord with all critic verdicts
verdict.json            # Final ValidationVerdict driving routing
final.json              # Assembled output
stages.log              # Per-stage: model, latency, token counts, retries
*_thinking.log          # Raw thinking block per stage — for debugging
sub_specs/{uuid}/       # Sub-spec run artefacts — decomposed tasks only
```

The database at `~/.pipeline/pipeline.db` stores a run index and accumulated
lessons. It lives outside the repo and is never committed.

---

## Running Tests

```bash
conda activate llama
cd ~/local-llama/Eolophus
pytest tests/unit/ -v
```

Unit tests require no model servers. They cover schema validation, guard logic,
routing functions, and lesson scoring arithmetic.

```bash
# Run a specific test file
pytest tests/unit/test_guards.py -v
pytest tests/unit/test_routers.py -v
pytest tests/unit/test_prefix.py -v
```
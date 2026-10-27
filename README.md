# Eolophus

Eolophus is a sophisticated, locally-run LLM orchestration pipeline designed for tackling complex tasks through a graph-based execution model. It features a modular architecture, multi-LLM support, a self-correction loop with a "lesson-learning" mechanism, and a dependency-free web UI for monitoring, interaction, and custom pipeline creation.

## Key Features

*   **Graph-Based Execution:** Utilizes LangGraph to define and execute complex task workflows as a stateful graph of interconnected nodes.
*   **Modular Node Architecture:** Tasks are broken down into discrete, reusable nodes, including classification, planning, drafting, appraisal, bug-fixing, and a multi-critic ensemble.
*   **Multi-LLM Strategy:** Assigns distinct roles to different models (e.g., Qwen for generation, DeepCoder for code analysis) and uses escalation ladders to dynamically switch to more powerful models when tasks prove difficult.
*   **Self-Correction & Learning:** Employs a critique and validation loop to iteratively improve generated output. It distills successful corrections into "lessons" stored in a SQLite database to enhance future performance on similar tasks.
*   **Custom Pipelines:** A YAML-based definition schema and a powerful web-based visual editor allow you to create and manage custom execution graphs beyond the built-in flow.
*   **CLI and Web Interface:** Interact with the pipeline via a comprehensive CLI (`run.py`) for scriptable execution or a dependency-free web UI for submitting tasks, monitoring live progress, managing models, and building pipelines.
*   **Integrated Capabilities:** Includes built-in support for vision (analyzing image inputs) and web search (using a local SearXNG instance) to ground its responses and analysis.
*   **Observability:** Integrates with Langfuse for detailed tracing and observability of pipeline runs.

## Architecture Overview

The repository is structured into several key directories:

*   `api/`: Contains the FastAPI server, all API endpoint logic, and the static assets for the web UI.
*   `clients/`: Provides wrappers for interacting with LLMs (`llm.py`), managing model server processes (`model_manager.py`), and external tools like web search (`search.py`).
*   `config/`: Home to all declarative configuration, including model definitions (`models.yaml`), pipeline routing logic (`routing.yaml`), and prompt templates for each node (`prompts/`).
*   `nodes/`: Implements the individual processing steps (nodes) of the pipeline, such as `classify_node`, `plan_node`, and `draft_node`.
*   `pipeline/`: Defines the LangGraph execution graph (`graph.py`), custom pipeline compiler (`custom_graph.py`), routing functions, and core state management.
*   `schemas/`: Contains all Pydantic models that define the data structures passed between nodes.
*   `storage/`: Manages the SQLite database for storing run history, critique records, and learned lessons.

## Getting Started

### 1. Installation

Clone the repository and install the required Python dependencies.

```bash
git clone https://github.com/mumflrr/eolophus.git
cd eolophus
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and customize it for your local setup.

```bash
cp .env.example .env
```

You will need to edit `.env` to specify `MODEL_DIR`, which is the directory where your GGUF model files are stored. You may also configure the database path and Langfuse credentials.

### 3. Database Initialization

The pipeline uses a SQLite database to store its state. The database and its schema will be created automatically the first time you run the application (either via the CLI or the web server).

## Usage

You can interact with the Eolophus pipeline through its command-line interface or the web UI.

### Command-Line Interface

The primary entry point is `run.py`. It accepts a task description and numerous flags to control its behavior.

**Basic Usage:**
```bash
python run.py "implement a FastAPI endpoint that returns the current time"
```

**Specifying Modes and Task Types:**
```bash
# Force the long, more thorough pipeline mode
python run.py --mode long "design a multi-tenant authentication system"

# Provide an image for vision analysis
python run.py --image ./mockup.png "implement this UI in HTML and CSS"
```

**Inline Prefix Syntax:**
You can also use a shorthand prefix syntax directly in your task description to control the pipeline's behavior. CLI flags will always take precedence.

*   `/long <task>`: Force long mode.
*   `/short <task>`: Force short mode.
*   `/no-ensemble <task>`: Skip the critique ensemble for a faster run.
*   `/long/coding <task>`: Force both long mode and the "coding" task type.

**Example with Prefix:**
```bash
python run.py "/long/ideation explore different approaches for a real-time collaboration tool"
```

### Web UI

The web UI provides a rich interface for submitting and monitoring runs, managing models, and creating custom pipelines.

1.  **Start the API Server:**
    ```bash
    uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
    ```

2.  **Access the UI:**
    Open your browser and navigate to `http://localhost:8000`.

The UI is a dependency-free, pure HTML/CSS/JS application that communicates with the FastAPI backend. It allows you to:
*   Submit new tasks with attachments and configuration options.
*   Watch the progress of a run in real-time through a stage-by-stage timeline.
*   Interact with a run that has paused for clarification.
*   Browse run artifacts, chat history, and learned lessons.
*   Manage model configurations and assign roles.
*   Create, edit, and delete custom pipelines with a visual node-graph editor.

## Configuration

Eolophus is highly configurable through YAML files located in the `config/` directory.

*   `models.yaml`: Defines all available LLMs, their server configurations (ports, launch flags), and their assigned roles within the pipeline (e.g., which model handles `classify` vs. `appraise`). It also specifies escalation ladders for each role.
*   `routing.yaml`: Controls the pipeline's routing logic, including thinking token budgets for each stage, correction loop limits, and triggers for the critique ensemble.
*   `prompts/`: Contains a YAML file for each node, defining the system prompt and user message templates used for that stage.

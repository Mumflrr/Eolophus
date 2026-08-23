# Eolophus Pipeline GUI

A dependency-free web GUI for the Eolophus pipeline API — no npm install,
no build step. Plain HTML/CSS/JS (ES modules), drops straight into
`api/static/`.

## Run it

```
pip install fastapi uvicorn[standard]
uvicorn api.mock_server:app --reload
```

Open `http://localhost:8000`. When real backend access is available, stop
`mock_server` and run `uvicorn api.server:app --reload` instead — the
frontend needs zero changes, since both servers expose the identical
`/pipelines/*` and other contracts.

If you run the GUI from somewhere other than the FastAPI server itself,
open **Settings** and change the API base URL (stored in `localStorage`,
defaults to `http://localhost:8000`).

## Layout

```
api/
  mock_server.py     ← now includes /pipelines/* CRUD (see below) — replace
                        your copy with this one, or diff and merge by hand
  static/
    index.html
    css/
      tokens.css        design tokens
      base.css           reset + app shell + icon baseline
      components.css      card, buttons, badges, forms, connection pill, toasts
      timeline.css         run-detail stage timeline, run-list rows, lesson cards
      pipelineEditor.css    the pipeline canvas: nodes, edges, inspector panel
    js/
      api.js            one function per endpoint, including /pipelines/*
      app.js             shell + hash router
      pipelineLayout.js   pure layout engine for the canvas (unit-tested separately)
      format.js            duration/date/number formatting, status-color mapping
      icons.js              small hand-rolled outline icon set
      toast.js               toast notifications
      ambient.js               background gradient's Ultra-mode crossfade
      components/
        health.js             persistent connection indicator, polls /health
        quickRun.js            top-bar "start a run" pulse button
        card.js                 shared card/empty-state/loading markup
      views/
        runs.js               task submission (incl. pipeline picker) + live run list
        runDetail.js            live stage timeline via SSE, clarification, artifacts
        models.js                model list, load/unload, role reassignment
        pipelines.js              saved-pipeline list: browse/create/delete
        pipelineEditor.js          the node-graph canvas editor
        lessons.js                 lesson browser
        settings.js                 API base URL, SearXNG, config reload,
                                     built-in pipeline's stage budgets
```

No React, no bundler, no external JS dependencies at runtime — fetches
Space Grotesk / IBM Plex Sans / IBM Plex Mono from Google Fonts, falls
back to system fonts cleanly if offline.

## Design

Follows `design_reference.md`: pure black base, diagonal blue→purple
gradient wash (crossfading toward indigo/purple in Ultra mode), frosted-
glass cards, pill buttons, monospaced data. Icons are a small hand-rolled
outline set with one consistent baseline size and stroke weight — no
external icon library.

The pipeline canvas follows the handoff doc's explicit color guidance:
existing/freeform/decision node types each get a distinct left-border
accent (blue/green/purple), decision nodes lean into the purple
"analysis mode" accent as the most structurally interesting node type,
and loop-back edges render dashed, curved, and orange — visually distinct
from the solid blue forward-flow edges.

**Node positions are not part of the saved data model** — there's no x/y
field anywhere in `PipelineDefinition`. The canvas computes a fresh
layered auto-layout from graph structure every time (same reachability
walk the backend's own validator does), and nodes are draggable within a
session for readability, but dragging is never persisted. That's a
deliberate choice to avoid inventing client-side state the server has
nowhere to keep — flagging it explicitly rather than silently faking
position persistence.

## Testing

Everything was exercised against a **live** `mock_server.py` — not mocked
responses — using jsdom + real `fetch` + real `EventSource`:

- **53-check full regression suite** covering every view: submission,
  live SSE stage streaming with no duplicate nodes, the clarification
  pause/resume cycle, model role reassignment, budget persistence
  (verified server-side), lesson deletion (verified server-side), run
  cancellation, and routing/unmount safety.
- **30-check pipeline editor suite**: create a pipeline, add all three
  step types, wire a decision's outcomes, trigger the loop-cap prompt on
  a backward drag exactly as the handoff doc asked for, live debounced
  validation reaching a clean "valid" state, save, reload from the server
  and confirm round-trip fidelity, run a real execution against the
  custom pipeline, delete.
- **15 unit tests** for the pure layout engine (`pipelineLayout.js`)
  against the handoff doc's own `test_loop` example, plus edge cases:
  orphaned/unreachable steps, dangling edge targets, self-loops, and
  backward `edge_overrides`.

## A real bug found in the backend, not the mock

While testing loop caps against the handoff doc's own `test_loop` example,
I found a genuine logic bug in `pipeline/custom_nodes.py`'s
`make_decision_router`, in the `if count >= cap:` block. Once a decision
step's visit count reaches its cap, the router force-reroutes to a
*different* outcome than whatever the model actually decided —
**unconditionally**, without first checking whether the raw decision
already exits the loop. So if a loop's model correctly decides "pass" on
exactly the visit that happens to also be the capped one, the cap logic
still overrides it and forces the *looping* branch instead — backward
from the intent. It only behaves correctly when the raw decision at the
cap boundary happens to already be the looping one, which is the common
case but not guaranteed.

I ported the exact current (buggy) behavior into the mock's
custom-pipeline simulator (`_simulate_custom_pipeline_run` in
`mock_server.py`), since the mock's job is to faithfully mirror what
`server.py` actually does — not a hypothetical fixed version. If this
gets fixed in the real backend, update the mock's simulator to match.

## Two earlier mock quirks (`mock_server.py`, non-blocking)

Neither is a contract violation, both worth knowing:

1. **`completed_at` never gets set** on a finished built-in-pipeline run —
   stays `null` forever. The run list falls back to `total_latency_ms`
   for duration, so this doesn't show up in the UI.
2. **Role reassignment doesn't actually change what `GET /models`
   returns afterward** in the original (non-pipeline) mock fixtures — the
   GUI works around this with an optimistic client-side layer that's
   harmless against the real `server.py` too.

Nothing in `server.py` itself needed changing for either.

## Custom pipelines — what's genuinely done vs. not

Per the handoff doc's own caveat, model-output quality on a real creative
pipeline with live models hasn't been tested by anyone — the backend
graph mechanics were verified with model calls stubbed out. The GUI side
is fully tested against the mock's simulated stage progression, but
obviously can't verify real inference quality either. `api_contract.md`
also still doesn't document `/pipelines/*` — this repo's `api.js` and
this README are the interim reference until that's updated.

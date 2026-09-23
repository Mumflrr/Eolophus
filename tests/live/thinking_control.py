#!/usr/bin/env python3
"""
tests/live/thinking_control.py — which thinking switch does YOUR llama-server actually honour?

NEEDS A LIVE llama-server (already running). Not part of `pytest`.

The pipeline marks stages like classify / validate / audit / bugfix "non-thinking", but if the server
ignores the switch they think anyway (invisibly, in `reasoning_content`), burn hundreds of tokens on a
~40-token JSON, and can run away entirely on a long prompt. This sends the same trivial question five
ways and reports how many tokens each variant generated.

    python tests/live/thinking_control.py                    # model/URL from config/models.yaml (9b)
    python tests/live/thinking_control.py --model 35b
    python tests/live/thinking_control.py --base-url http://127.0.0.1:8081/v1 --model-id whatever

Needs the server running. Takes a few seconds per variant (the baseline may think for a while).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

PROMPT = "Is 17 a prime number? Answer with exactly one word: yes or no."

# (label, extra request fields). Order matters: the analysis compares later variants against the baseline.
VARIANTS = [
    ("baseline (nothing sent)",                       {}),
    ("`thinking` disabled  (what the pipeline sent before)", {"thinking": {"type": "disabled"}}),
    ("chat_template_kwargs.enable_thinking=false",    {"chat_template_kwargs": {"enable_thinking": False}}),
    ("reasoning_budget=0",                            {"reasoning_budget": 0}),
    ("both `thinking` + chat_template_kwargs (now)",  {"thinking": {"type": "disabled"},
                                                       "chat_template_kwargs": {"enable_thinking": False}}),
]
OFF_THRESHOLD = 40          # a one-word answer is ~2 tokens; anything past this is hidden thinking


def find_project_root(start: pathlib.Path):
    for c in (start, *start.parents):
        if (c / "clients").is_dir() and (c / "schemas").is_dir():
            return c
    return None


def probe(post, base_url: str, model_id: str, extra: dict, max_tokens: int) -> dict:
    """One request. `post(url, json) -> dict` is injected so this is testable without a server."""
    body = {"model": model_id, "max_tokens": max_tokens, "temperature": 0.6,
            "messages": [{"role": "user", "content": PROMPT}], **extra}
    data = post(f"{base_url.rstrip('/')}/chat/completions", body)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if "<think>" in content:                                   # thinking delivered inline instead
        inline = content.split("<think>", 1)[1].split("</think>", 1)[0]
        reasoning = reasoning or inline
    return {"tokens": (data.get("usage") or {}).get("completion_tokens", 0), "reasoning_chars": len(reasoning),
            "content": content.split("</think>")[-1].strip()[:40],
            "truncated": data["choices"][0].get("finish_reason") == "length"}


def is_off(r: dict) -> bool:
    return r["tokens"] <= OFF_THRESHOLD and r["reasoning_chars"] == 0


def analyze(results: list[tuple[str, dict]]) -> list[str]:
    """Plain-English conclusions from the five probes."""
    by = {label: r for label, r in results}
    base, legacy, ctk, budget, both = (r for _, r in results)
    out = []
    if is_off(base):
        return ["The baseline request already produces no thinking, so hidden thinking is NOT what is making your "
                "stages long. Look elsewhere (the prompt, the model's answer length, or the output cap)."]
    if is_off(legacy):
        out.append("`thinking: disabled` DOES work on this server — the legacy switch is fine.")
    else:
        out.append("`thinking: disabled` is IGNORED by this server (it still thinks) — this is why 'non-thinking' "
                   "stages were thinking anyway.")
    if is_off(ctk):
        out.append("chat_template_kwargs.enable_thinking=false WORKS. Keep routing.yaml -> thinking_control."
                   "chat_template_kwargs: true (the default) so every non-thinking stage really is non-thinking.")
    elif is_off(budget):
        out.append("Only reasoning_budget=0 works here. The pipeline's helper sends that alongside for thinking budgets, "
                   "but for non-thinking stages you would need chat_template_kwargs to work too — tell me and I'll "
                   "send reasoning_budget=0 for them as well.")
    elif not is_off(both):
        out.append("NONE of the per-request switches turn thinking off on this server. Set it server-side instead: "
                   "start llama-server with --reasoning-budget 0 for a non-thinking instance (or, if your build has "
                   "it, --chat-template-kwargs '{\"enable_thinking\":false}'), e.g. via config/llama_flags/9b.sh.")
    if any(r["truncated"] for r in results_values(results)):
        out.append("At least one variant hit max_tokens while thinking — that is the runaway, reproduced.")
    return out


def results_values(results):
    return [r for _, r in results]


def http_post(url: str, body: dict) -> dict:
    import httpx
    resp = httpx.post(url, json=body, timeout=600.0)
    resp.raise_for_status()
    return resp.json()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="9b", help="models.yaml key to read base_url/model_id from (default 9b)")
    ap.add_argument("--base-url"), ap.add_argument("--model-id")
    ap.add_argument("--max-tokens", type=int, default=1500, help="cap per probe (default 1500)")
    args = ap.parse_args(argv)

    base_url, model_id = args.base_url, args.model_id
    if not (base_url and model_id):
        root = find_project_root(pathlib.Path(__file__).resolve().parent)
        if root is None:
            print("Couldn't find the project root to read config/models.yaml — pass --base-url and --model-id.", file=sys.stderr)
            return 2
        sys.path.insert(0, str(root))
        from clients.llm import get_model_config
        cfg = get_model_config(args.model)
        base_url, model_id = base_url or cfg["base_url"], model_id or cfg["model_id"]

    print(f"Probing {base_url}  (model {model_id!r}, cap {args.max_tokens} tokens)\n")
    results = []
    for label, extra in VARIANTS:
        try:
            r = probe(http_post, base_url, model_id, extra, args.max_tokens)
        except Exception as e:
            print(f"Could not reach the server for {label!r}: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        results.append((label, r))
        state = "thinking OFF" if is_off(r) else "THINKING ON"
        print(f"  {label:55s} {r['tokens']:5d} tokens  {r['reasoning_chars']:6d} chars reasoning  "
              f"[{state}]{'  HIT CAP' if r['truncated'] else ''}  -> {r['content']!r}")
    print()
    for line in analyze(results):
        print(" *", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
test_tool_calling.py — standalone probe for native tool-calling support.

Sends ONE raw chat.completions.create(tools=[...]) request directly to a
running llama.cpp server (bypassing clients/llm.py, instructor, and the
whole pipeline entirely) and prints exactly what comes back.

This answers one question only: does THIS model, on THIS llama.cpp build,
with THIS chat template, actually populate response.choices[0].message.tool_calls
when given a tools= schema and a prompt that should obviously trigger a
tool call — or does it ignore `tools` and just answer as plain text
(or worse, hallucinate a fake "call" as prose inside message.content
instead of a real structured tool_calls entry)?

Usage:
    python test_tool_calling.py                # tests port 8081 (9B)
    python test_tool_calling.py --port 8083     # tests 35B instead
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current, up-to-date information. Use this whenever the user asks about something that requires current/real-time data, such as today's date, current events, or recent facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short keyword search query, 3-8 words, like what a person would type into a search box.",
                },
            },
            "required": ["query"],
        },
    },
}

# Deliberately impossible for the model to answer without calling the
# tool — no training data has "today's date" or "today's top headline",
# so a model that's ignoring `tools` entirely should either refuse,
# hallucinate an answer, or ask a clarifying question — NOT produce a
# real tool_calls entry. A model that DOES support tool calling should
# emit tool_calls with a sensible query and empty/near-empty content.
PROMPT = "What is today's top news headline? Use the search_web tool to find out."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081, help="llama.cpp server port (default 8081 = 9B)")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    print(f"=== Testing tool-calling support at {base_url} ===\n")

    client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=60.0)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": PROMPT},
    ]

    print("Sending request with tools=[search_web]...")
    print(f"Prompt: {PROMPT!r}\n")

    try:
        resp = client.chat.completions.create(
            model="local",  # llama.cpp server ignores this field, but SDK requires it
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice="auto",
            temperature=0.6,
        )
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
        print("\nThis likely means the server rejected the `tools` parameter outright —")
        print("check that llama.cpp was started with --jinja and is a recent-enough build")
        print("to parse tool-call output for this model's chat template.")
        sys.exit(1)

    msg = resp.choices[0].message
    finish_reason = resp.choices[0].finish_reason

    print(f"finish_reason: {finish_reason}")
    print(f"message.content: {msg.content!r}")
    print(f"message.tool_calls: {msg.tool_calls!r}\n")

    if msg.tool_calls:
        print("=== RESULT: Native tool calling WORKS ===")
        for tc in msg.tool_calls:
            print(f"  tool: {tc.function.name}")
            print(f"  arguments (raw): {tc.function.arguments!r}")
            try:
                parsed = json.loads(tc.function.arguments)
                print(f"  arguments (parsed): {parsed}")
            except json.JSONDecodeError as e:
                print(f"  WARNING: arguments did not parse as JSON: {e}")
        print("\nThe tool_calls field is populated with structured data — this is a real")
        print("native tool call, not prose. Safe to build call_model_with_tools() on this.")
    else:
        print("=== RESULT: No tool_calls returned ===")
        print("The model answered as plain text instead of calling the tool, or the")
        print("server silently dropped/ignored the `tools` parameter. Check:")
        print("  1. Does message.content look like it's IMITATING a tool call in prose")
        print("     (e.g. 'Action: search_web...') rather than using structured tool_calls?")
        print("     If so, this model/template needs a ReAct-style text protocol instead")
        print("     of native tools=, at least for this role.")
        print("  2. Try again — Qwen3 models are sometimes inconsistent about invoking")
        print("     tools on the first attempt depending on temperature/sampling.")
        print("  3. Confirm llama.cpp server startup logs mention loading a tool-call")
        print("     parser for this model (grep server logs for 'tool' at startup).")

    print(f"\nFull raw response object (for debugging):")
    print(resp.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
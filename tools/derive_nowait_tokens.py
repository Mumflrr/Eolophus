#!/usr/bin/env python3
"""
tools/derive_nowait_tokens.py — empirically derive NoWait token IDs.

The NoWait paper shows that suppressing reflection tokens (Wait, Hmm,
Alternatively, However...) reduces CoT length by 27-51% with no quality loss.
Token IDs are model-specific — this tool derives them for your specific model.

Method:
  1. Send N diverse reasoning prompts to the model with thinking mode ON
  2. Collect all <think> block tokens
  3. Find the most frequent tokens that are reflection phrases
  4. Output token IDs for pasting into config/models.yaml

Usage:
  python tools/derive_nowait_tokens.py --model 35b --samples 32
  python tools/derive_nowait_tokens.py --model 9b --samples 16 --port 8081

After running, paste the output into config/models.yaml under nowait_tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Reflection phrases to look for ───────────────────────────────────────────
# These are the phrases identified in the NoWait paper as causing redundant
# backtracking in reasoning models. We search for their token representations.

REFLECTION_PHRASES = [
    "Wait",   "wait",
    "Hmm",    "hmm",   "Hm",  "hm",
    "Actually", "actually",
    "However", "however",
    "Alternatively", "alternatively",
    "But wait", "But Wait",
    "Let me reconsider", "let me reconsider",
    "On second thought", "on second thought",
    "Actually,", "actually,",
    "Wait,", "wait,",
    "Hmm,", "hmm,",
    "However,", "however,",
    "Check", "check",
    "Double-check", "double-check",
    "Re-check", "re-check",
    "Let me check", "let me check",
    "Another approach", "another approach",
    "Another way", "another way",
]

# Diverse reasoning prompts to elicit thinking tokens
SAMPLE_PROMPTS = [
    "Write a Python function to find the nth Fibonacci number efficiently.",
    "Design a simple REST API for a todo list application.",
    "Explain the tradeoffs between SQL and NoSQL databases.",
    "How would you implement a rate limiter in Python?",
    "Write a function to detect if a linked list has a cycle.",
    "Design a caching layer for a web application.",
    "How would you handle concurrent writes to a shared resource?",
    "Write a function to parse a CSV file with error handling.",
    "Implement a simple pub/sub system in Python.",
    "How would you implement pagination in a REST API?",
    "Write a binary search implementation with proper edge cases.",
    "Design a simple task queue with worker processes.",
    "How would you implement JWT authentication?",
    "Write a decorator for retry logic with exponential backoff.",
    "Implement a simple LRU cache.",
    "How would you structure a multi-tenant database schema?",
    "Write a function to validate an email address.",
    "Implement a simple state machine.",
    "How would you design a URL shortener?",
    "Write a function to find all permutations of a string.",
    "Implement a simple event system in Python.",
    "How would you handle database migrations?",
    "Write a function to deep merge two dictionaries.",
    "Implement a simple circuit breaker pattern.",
    "How would you design a leaderboard system?",
    "Write a function to flatten a nested list.",
    "Implement a simple dependency injection container.",
    "How would you implement request deduplication?",
    "Write a function to convert between time zones.",
    "Implement a simple work stealing scheduler.",
    "How would you design a distributed lock?",
    "Write a function to calculate edit distance between strings.",
]


def collect_thinking_tokens(
    base_url: str,
    model_id: str,
    n_samples: int,
    budget_tokens: int = 2048,
) -> list[str]:
    """
    Collect thinking block text from N diverse prompts.
    Returns a flat list of tokens/words from all think blocks.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="local")
    prompts = SAMPLE_PROMPTS[:n_samples]
    all_thinking_text = []

    print(f"Collecting {n_samples} samples from {base_url}...")

    for i, prompt in enumerate(prompts, 1):
        try:
            resp = client.chat.completions.create(
                model       = model_id,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.7,
                extra_body  = {
                    "thinking": {
                        "type":          "enabled",
                        "budget_tokens": budget_tokens,
                    }
                },
            )
            content = resp.choices[0].message.content or ""

            # Extract thinking block
            match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            if match:
                think_text = match.group(1)
                all_thinking_text.append(think_text)
                print(f"  [{i}/{n_samples}] collected {len(think_text)} chars of thinking")
            else:
                print(f"  [{i}/{n_samples}] no thinking block found")

        except Exception as e:
            print(f"  [{i}/{n_samples}] ERROR: {e}")
            continue

    return all_thinking_text


def find_reflection_tokens(thinking_texts: list[str]) -> dict[str, int]:
    """
    Count how often each reflection phrase appears across all thinking texts.
    Returns {phrase: count} sorted by frequency.
    """
    counts: Counter = Counter()

    for text in thinking_texts:
        # Count each phrase at start of a word boundary
        for phrase in REFLECTION_PHRASES:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            matches = re.findall(pattern, text)
            if matches:
                counts[phrase] += len(matches)

    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def get_token_ids(
    base_url: str,
    model_id: str,
    phrases: list[str],
) -> dict[str, list[int]]:
    """
    Attempt to get token IDs for phrases using the tokenise endpoint.
    Falls back to placeholder if endpoint unavailable.

    Note: llama.cpp server exposes /tokenize at the base path (not /v1/tokenize).
    """
    import urllib.request
    import urllib.error

    # Try llama.cpp tokenize endpoint
    tokenize_url = base_url.replace("/v1", "") + "/tokenize"
    phrase_to_ids: dict[str, list[int]] = {}

    for phrase in phrases:
        try:
            data = json.dumps({"content": phrase, "add_special": False}).encode()
            req  = urllib.request.Request(
                tokenize_url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
                tokens = result.get("tokens", [])
                if tokens:
                    phrase_to_ids[phrase] = tokens
        except Exception:
            # Endpoint not available — return placeholder
            phrase_to_ids[phrase] = [-1]  # placeholder

    return phrase_to_ids


def main():
    parser = argparse.ArgumentParser(
        description="Derive NoWait token IDs empirically for a model"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["9b", "27b", "35b", "deepcoder", "coder14b"],
        help="Model to analyse",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Number of reasoning samples to collect (default: 32)",
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=2048,
        dest="budget_tokens",
        help="Thinking token budget per sample (default: 2048)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port (reads from models.yaml by default)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        dest="top_n",
        help="Number of top reflection phrases to output (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any API calls",
    )

    args = parser.parse_args()

    # Load model config
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["models"][args.model]
    port      = args.port or model_cfg["port"]
    base_url  = f"http://localhost:{port}/v1"
    model_id  = model_cfg["model_id"]

    print(f"\nNoWait Token Derivation")
    print(f"  Model:    {model_cfg['name']} ({args.model})")
    print(f"  Port:     {port}")
    print(f"  Samples:  {args.samples}")
    print(f"  Budget:   {args.budget_tokens} thinking tokens")
    print()

    if args.dry_run:
        print("[DRY RUN] Would sample these prompts:")
        for i, p in enumerate(SAMPLE_PROMPTS[:args.samples], 1):
            print(f"  {i}. {p[:70]}...")
        return

    # Step 1: Collect thinking text
    thinking_texts = collect_thinking_tokens(base_url, model_id, args.samples, args.budget_tokens)

    if not thinking_texts:
        print("\nERROR: No thinking text collected. Is the model server running?")
        sys.exit(1)

    total_chars = sum(len(t) for t in thinking_texts)
    print(f"\nCollected {len(thinking_texts)} thinking blocks ({total_chars:,} chars total)")

    # Step 2: Find reflection phrases
    phrase_counts = find_reflection_tokens(thinking_texts)

    if not phrase_counts:
        print("\nNo reflection phrases found. Model may not use these patterns.")
        sys.exit(0)

    print(f"\nTop reflection phrases found:")
    top_phrases = list(phrase_counts.items())[:args.top_n]
    for phrase, count in top_phrases:
        pct = count / len(thinking_texts) * 100
        print(f"  {phrase!r:30s} → {count:4d} occurrences ({pct:.0f}% of samples)")

    # Step 3: Get token IDs
    top_phrase_list = [p for p, _ in top_phrases if phrase_counts[p] > 1]
    print(f"\nFetching token IDs for {len(top_phrase_list)} phrases...")
    phrase_to_ids = get_token_ids(base_url, model_id, top_phrase_list)

    # Step 4: Output YAML block
    print(f"\n{'='*60}")
    print(f"Add this to config/models.yaml under models.{args.model}.nowait_tokens:")
    print(f"{'='*60}")
    print(f"    nowait_tokens:")

    has_real_ids = False
    for phrase, count in top_phrases:
        ids = phrase_to_ids.get(phrase, [-1])
        if ids and ids[0] != -1:
            has_real_ids = True
            for tid in ids:
                print(f"      {tid}: -100.0  # '{phrase}' (seen {count}x)")

    if not has_real_ids:
        print("      # Token IDs unavailable — tokenize endpoint not accessible")
        print("      # Run with a running model server to get actual token IDs")
        print("      # Or use llama.cpp's built-in --special-tokens-file approach")
        print()
        print("      # Phrases to suppress (manually find IDs with your tokeniser):")
        for phrase, count in top_phrases[:10]:
            print(f"      # '{phrase}' seen {count}x")

    print(f"{'='*60}")
    print()
    print("NOTE: Apply NoWait to the 35B planning pass only.")
    print("Set budget_tokens to limit thinking depth, not NoWait alone.")


if __name__ == "__main__":
    main()

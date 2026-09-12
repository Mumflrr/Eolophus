#!/usr/bin/env python3
"""
debug_search.py — run this ON YOUR MACHINE (not here — I have no network
access in this sandbox) to isolate exactly where the search chain breaks.

Usage:
    python debug_search.py
    python debug_search.py "custom query"
"""
import os
import sys

import httpx

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
query = sys.argv[1] if len(sys.argv) > 1 else "test query"

print(f"[1] SEARXNG_URL = {SEARXNG_URL}")
print(f"    (this is what GET /config/searxng on your running server should also show —")
print(f"     if they differ, the server process has a different env than this shell)")
print()

print(f"[2] Checking base reachability: {SEARXNG_URL}")
try:
    r = httpx.get(SEARXNG_URL, timeout=5.0)
    print(f"    -> HTTP {r.status_code} (server is up)")
except httpx.ConnectError as e:
    print(f"    -> CONNECTION FAILED: {e}")
    print(f"    SearXNG isn't reachable at all. Check it's running: docker ps / systemctl status")
    sys.exit(1)
except Exception as e:
    print(f"    -> unexpected error: {e}")
    sys.exit(1)
print()

print(f"[3] Checking JSON API: {SEARXNG_URL}/search?q={query}&format=json")
try:
    r = httpx.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"}, timeout=10.0)
    print(f"    -> HTTP {r.status_code}")
    if r.status_code == 403:
        print()
        print("    *** THIS IS LIKELY YOUR BUG ***")
        print("    SearXNG disables the JSON output format by default.")
        print("    Fix: edit your SearXNG settings.yml, find:")
        print("        search:")
        print("          formats:")
        print("            - html")
        print("    and add:")
        print("            - json")
        print("    then restart SearXNG (docker restart <container>, or systemctl restart searxng).")
        sys.exit(1)
    r.raise_for_status()
except httpx.HTTPStatusError as e:
    print(f"    -> HTTP error: {e}")
    sys.exit(1)
print()

print("[4] Parsing response as JSON")
try:
    data = r.json()
except ValueError as e:
    print(f"    -> NOT VALID JSON: {e}")
    print(f"    First 300 chars of body: {r.text[:300]!r}")
    print("    This usually means format=json isn't actually enabled server-side")
    print("    even though the request didn't 403 (some SearXNG versions return")
    print("    an HTML error page with 200 status instead of 403).")
    sys.exit(1)

results = data.get("results", [])
print(f"    -> Got {len(results)} result(s)")
if results:
    print(f"    First result: {results[0].get('title')!r} -> {results[0].get('url')!r}")
print()

print("[5] Chain summary")
print("    SearXNG itself is reachable and returning usable JSON results.")
print("    If web search STILL doesn't show up in pipeline output, the problem")
print("    is downstream of this — i.e. no node is calling search_web() at all.")
print("    Check: does nodes/planner.py or nodes/ideation.py import")
print("    clients.search and check state.get('use_search')? Search for it:")
print("        grep -rn 'search_web\\|use_search' nodes/")
print("    If that grep comes back empty, that's the actual remaining gap —")
print("    the HTTP client works, but nothing in the pipeline calls it yet.")
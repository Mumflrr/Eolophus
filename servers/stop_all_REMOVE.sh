#!/bin/bash
# stop_all.sh — stop all llama.cpp server instances cleanly
# SearXNG (Docker container, started by start_all.sh's start_searxng) is
# stopped separately below — it has no .pid file to iterate over here since
# Docker manages its own process, not this script.

LOG_DIR="$HOME/pipeline/logs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Stopping pipeline servers ==="

for pidfile in "$LOG_DIR"/*.pid; do
    [ -f "$pidfile" ] || continue
    name=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping $name (PID $pid)..."
        kill "$pid"
        rm "$pidfile"
    else
        echo "  $name not running (stale PID $pid)"
        rm "$pidfile"
    fi
done

echo ""
echo "=== Stopping SearXNG ==="
if ! command -v docker &> /dev/null; then
    echo "  Docker not found — nothing to stop."
elif [ ! -f "$SCRIPT_DIR/docker-compose.searxng.yml" ]; then
    echo "  docker-compose.searxng.yml not found next to stop_all.sh — skipping."
else
    # Mirrors start_all.sh's start_searxng: doesn't fail the whole script
    # if this errors (e.g. it was never started this session) — same
    # non-fatal treatment as SearXNG gets on the start side.
    (cd "$SCRIPT_DIR" && docker-compose -f docker-compose.searxng.yml down) || \
        echo "  (SearXNG container wasn't running, or already stopped)"
fi

echo ""
echo "=== All servers stopped ==="
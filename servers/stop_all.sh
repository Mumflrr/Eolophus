#!/bin/bash
# stop_all.sh — stop all llama.cpp server instances cleanly

LOG_DIR="$HOME/pipeline/logs"

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

echo "=== All servers stopped ==="

#!/bin/bash
# start_all.sh — start all llama.cpp server instances in dependency order
# Each server is health-checked before the script exits.
# 9B starts first (used in every stage); 27B starts last (long mode only).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/pipeline/logs"
mkdir -p "$LOG_DIR"

MAX_WAIT=120  # seconds per server health check

health_check() {
    local port=$1
    local name=$2
    local elapsed=0
    echo -n "  Waiting for $name (port $port)..."
    while ! curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo " TIMEOUT after ${MAX_WAIT}s"
            return 1
        fi
        echo -n "."
    done
    echo " OK (${elapsed}s)"
    return 0
}

start_server() {
    local script=$1
    local name=$2
    local port=$3
    echo "Starting $name..."
    bash "$SCRIPT_DIR/config/llama_flags/$script" &
    echo $! > "$LOG_DIR/${name}.pid"
    health_check $port $name || {
        echo "ERROR: $name failed to start. Check $LOG_DIR/${name}_server.log"
        exit 1
    }
}

echo "=== Starting pipeline servers ==="
start_server "9b.sh"        "9b"        8081
start_server "35b.sh"       "35b"       8083
start_server "deepcoder.sh" "deepcoder" 8084
start_server "coder14b.sh"  "coder14b"  8085
start_server "27b.sh"       "27b"       8082

echo ""
echo "=== All servers running ==="
echo "  9B:        http://localhost:8081"
echo "  27B:       http://localhost:8082"
echo "  35B:       http://localhost:8083"
echo "  DeepCoder: http://localhost:8084"
echo "  Coder14B:  http://localhost:8085"
echo ""
echo "Run 'bash servers/stop_all.sh' to stop all servers."

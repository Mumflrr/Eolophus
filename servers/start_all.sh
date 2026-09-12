#!/bin/bash
# start_all.sh — start all llama.cpp server instances in dependency order
# Each server is health-checked before the script exits.
# 9B starts first (used in every stage); 27B starts last (long mode only).
# SearXNG (Docker container) starts alongside them — see docker-compose.searxng.yml.

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

# SearXNG isn't a llama.cpp process like the others — it's a Docker
# container (see docker-compose.searxng.yml / setup_searxng.sh) — so it
# gets its own start function rather than reusing start_server, which
# assumes a backgrounded `bash script.sh &` + a .pid file. Docker manages
# the container's own lifecycle; there's no PID for us to track here.
# health_check works unmodified since SearXNG's health check is the same
# "does /health respond" shape as the model servers, on its own port.
start_searxng() {
    local port=8888
    if ! command -v docker &> /dev/null; then
        echo "WARNING: Docker not found — skipping SearXNG (web search will be unavailable)."
        echo "  Run ./setup_searxng.sh once to install Docker and SearXNG, then re-run this script."
        return 0
    fi
    if [ ! -f "$SCRIPT_DIR/docker-compose.searxng.yml" ]; then
        echo "WARNING: docker-compose.searxng.yml not found next to start_all.sh — skipping SearXNG."
        return 0
    fi

    echo "Starting SearXNG..."
    (cd "$SCRIPT_DIR" && docker-compose -f docker-compose.searxng.yml up -d)

    # SearXNG doesn't expose a /health endpoint — probe the base URL instead,
    # same idea as health_check but without assuming that path exists.
    local elapsed=0
    echo -n "  Waiting for searxng (port $port)..."
    while ! curl -sf "http://localhost:$port/" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo " TIMEOUT after ${MAX_WAIT}s"
            echo "  ERROR: SearXNG failed to start. Check: docker-compose -f docker-compose.searxng.yml logs"
            # Not exiting here (unlike start_server's health_check failure) —
            # web search is opt-in per-run (use_search), so a broken SearXNG
            # shouldn't block the whole pipeline from starting.
            return 1
        fi
        echo -n "."
    done
    echo " OK (${elapsed}s)"
}

echo "=== Starting pipeline servers ==="
start_server "9b.sh"        "9b"        8081
start_server "35b.sh"       "35b"       8083
start_server "deepcoder.sh" "deepcoder" 8084
start_server "coder14b.sh"  "coder14b"  8085
start_server "27b.sh"       "27b"       8082
start_searxng

echo ""
echo "=== All servers running ==="
echo "  9B:        http://localhost:8081"
echo "  27B:       http://localhost:8082"
echo "  35B:       http://localhost:8083"
echo "  DeepCoder: http://localhost:8084"
echo "  Coder14B:  http://localhost:8085"
echo "  SearXNG:   http://localhost:8888"
echo ""
echo "Run 'bash servers/stop_all.sh' to stop all servers."
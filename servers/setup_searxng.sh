#!/bin/bash
# setup_searxng.sh — ONE-TIME setup: installs Docker Engine (if missing)
# and brings up the SearXNG container defined in docker-compose.searxng.yml.
#
# Run this once, by hand, NOT from start_all.sh — start_all.sh should only
# ever START an already-installed service, the same way it doesn't install
# llama.cpp itself. This script is the install step that has to happen
# before start_all.sh's SearXNG block (see the patch to start_all.sh) has
# anything to start.
#
# Assumes Ubuntu (22.04/24.04/26.04) — see docs.docker.com/engine/install
# if you're on something else.
#
# Usage:
#   chmod +x setup_searxng.sh
#   ./setup_searxng.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 1. Checking for Docker ==="
if command -v docker &> /dev/null; then
    echo "Docker already installed: $(docker --version)"
else
    echo "Docker not found — installing Docker Engine from the official repo."
    echo "(This needs sudo. Steps taken from docs.docker.com/engine/install/ubuntu/)"

    sudo apt update
    sudo apt install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo ""
    echo "Docker installed. Verifying the daemon is running..."
    sudo systemctl enable --now docker

    # Let the current user run docker without sudo (post-install step from
    # docs.docker.com/engine/install/linux-postinstall). Requires a new
    # login shell / `newgrp docker` to take effect this session.
    if ! groups "$USER" | grep -q '\bdocker\b'; then
        sudo groupadd docker 2>/dev/null || true
        sudo usermod -aG docker "$USER"
        echo ""
        echo "Added $USER to the 'docker' group so you don't need sudo for docker/compose."
        echo "This won't take effect in your CURRENT shell — either log out and back in,"
        echo "or run 'newgrp docker' now, before continuing."
        echo ""
        read -p "Press Enter once you've done that (or Ctrl-C to do it yourself and re-run this script)..."
    fi
fi

echo ""
echo "=== 2. Starting SearXNG ==="
cd "$SCRIPT_DIR"

if [ ! -f docker-compose.searxng.yml ]; then
    echo "ERROR: docker-compose.searxng.yml not found in $SCRIPT_DIR"
    echo "Place it next to this script (same directory as start_all.sh) and re-run."
    exit 1
fi

mkdir -p searxng

FIRST_RUN=false
if [ ! -f searxng/settings.yml ]; then
    FIRST_RUN=true
fi

docker-compose -f docker-compose.searxng.yml up -d

echo -n "Waiting for SearXNG to come up..."
elapsed=0
while ! curl -sf "http://localhost:8888/" > /dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $elapsed -ge 60 ]; then
        echo " TIMEOUT after 60s — check 'docker-compose -f docker-compose.searxng.yml logs'"
        exit 1
    fi
    echo -n "."
done
echo " OK (${elapsed}s)"

if [ "$FIRST_RUN" = true ]; then
    echo ""
    echo "=== 3. First-run config: enabling JSON output ==="
    echo "SearXNG just generated ./searxng/settings.yml. The pipeline needs"
    echo "JSON search results, which SearXNG disables by default."
    echo ""
    echo "Stopping the container to edit settings.yml safely..."
    docker-compose -f docker-compose.searxng.yml down

    SETTINGS_FILE="searxng/settings.yml"
    if grep -q "^\s*- html\s*$" "$SETTINGS_FILE" && ! grep -q "^\s*- json\s*$" "$SETTINGS_FILE"; then
        # Insert '- json' right after the '- html' line under formats:
        sed -i '/^\s*- html\s*$/a\  - json' "$SETTINGS_FILE"
        echo "Added 'json' to search.formats in $SETTINGS_FILE"
    else
        echo "Could not automatically confirm/edit search.formats in $SETTINGS_FILE."
        echo "Please open it yourself, find:"
        echo "    search:"
        echo "      formats:"
        echo "        - html"
        echo "and add a line right after it:"
        echo "        - json"
    fi

    echo "Restarting with JSON output enabled..."
    docker-compose -f docker-compose.searxng.yml up -d
    sleep 3
fi

echo ""
echo "=== Done ==="
echo "SearXNG is running at http://localhost:8888"
echo "Verify the full chain with: python debug_search.py"
echo ""
echo "From now on, start/stop it the same way as your model servers:"
echo "  docker-compose -f docker-compose.searxng.yml up -d    # start"
echo "  docker-compose -f docker-compose.searxng.yml down     # stop"
echo "(start_all.sh / stop_all.sh can do this for you — see the patch)"
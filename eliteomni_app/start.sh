#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
#  EliteOmni v17 — Local Server Launcher
#  Fixed: SearXNG health verification + auto-heal before app starts
# ─────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

echo ""
echo " ========================================"
echo "  EliteOmni v17 - Starting Local Server"
echo " ========================================"
echo ""

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -f ~/eliteomni/bin/activate ]; then
    err "Virtual environment not found at ~/eliteomni — run: python -m venv ~/eliteomni"
    exit 1
fi
source ~/eliteomni/bin/activate
ok "Virtual environment activated"

# ── 2. Python check ───────────────────────────────────────────────────────────
if ! command -v python &> /dev/null; then
    err "Python not found in virtual environment."
    exit 1
fi
ok "Python: $(python --version 2>&1)"

# ── 3. Dependencies ───────────────────────────────────────────────────────────
echo "  Checking dependencies..."
pip install fastapi uvicorn faiss-cpu numpy requests --quiet --exists-action i
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    --quiet --exists-action i
ok "Dependencies OK"

# ── 4. SearXNG: ensure running BEFORE the app launches ───────────────────────
echo ""
echo "  Checking SearXNG..."

SEARXNG_PORT=8888
SEARXNG_URL="http://localhost:${SEARXNG_PORT}"

_searxng_probe() {
    # Try /healthz first, fall back to a test search
    curl -sf --max-time 3 "${SEARXNG_URL}/healthz" > /dev/null 2>&1 && return 0
    curl -sf --max-time 3 "${SEARXNG_URL}/search?q=test&format=json" > /dev/null 2>&1 && return 0
    return 1
}

_ensure_searxng() {
    # Already up?
    if _searxng_probe; then
        ok "SearXNG already running on :${SEARXNG_PORT}"
        return 0
    fi

    # Docker available?
    if ! command -v docker &> /dev/null; then
        warn "Docker not found — SearXNG cannot be auto-started."
        warn "Web search will be disabled until SearXNG is running."
        warn "Start it manually: docker run -d --name searxng --restart unless-stopped \\"
        warn "    -p ${SEARXNG_PORT}:8080 -e SEARXNG_SECRET_KEY=eliteomni searxng/searxng:latest"
        return 1
    fi

    # Container exists but stopped?
    CONTAINER_STATUS=$(docker inspect --format '{{.State.Status}}' searxng 2>/dev/null)
    if [ "$CONTAINER_STATUS" = "exited" ] || [ "$CONTAINER_STATUS" = "created" ]; then
        echo "  Starting existing SearXNG container..."
        docker start searxng > /dev/null 2>&1
    elif [ "$CONTAINER_STATUS" = "running" ]; then
        # Running but not answering — restart it
        warn "SearXNG container is running but not responding — restarting..."
        docker restart searxng > /dev/null 2>&1
    else
        # Container doesn't exist — create it fresh
        echo "  Creating SearXNG container for the first time..."
        docker run -d \
            --name searxng \
            --restart unless-stopped \
            -p "${SEARXNG_PORT}:8080" \
            -e SEARXNG_SECRET_KEY=eliteomni \
            searxng/searxng:latest > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            err "Failed to create SearXNG container."
            err "Try manually: docker run -d --name searxng -p ${SEARXNG_PORT}:8080 searxng/searxng:latest"
            return 1
        fi
    fi

    # Wait up to 20 s for SearXNG to come online
    echo "  Waiting for SearXNG to be ready..."
    for i in $(seq 1 20); do
        sleep 1
        if _searxng_probe; then
            ok "SearXNG ready on :${SEARXNG_PORT} (took ${i}s)"
            return 0
        fi
        printf "."
    done
    echo ""

    warn "SearXNG did not respond within 20s."
    warn "The app will start anyway — the watchdog will keep retrying in the background."
    warn "Debug: docker logs searxng"
    return 1
}

_ensure_searxng
SEARXNG_OK=$?

# ── 5. Export paths ───────────────────────────────────────────────────────────
export GGUF_MODEL_PATH="/mnt/c/Users/kidus yared/Downloads/qwen2.5-1.5b-instruct-q4_k_m.gguf"
export SEARXNG_URL="${SEARXNG_URL}"

# Sanity-check the model file exists
if [ ! -f "$GGUF_MODEL_PATH" ]; then
    warn "Model file not found at: $GGUF_MODEL_PATH"
    warn "Check the path in start.sh — GGUF_MODEL_PATH"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo " ┌─────────────────────────────────────────────┐"
printf " │  Model:  %-35s│\n" "$(basename "$GGUF_MODEL_PATH")"
printf " │  App:    %-35s│\n" "http://localhost:8080"
printf " │  LAN:    %-35s│\n" "http://$(hostname -I | awk '{print $1}'):8080"
if [ $SEARXNG_OK -eq 0 ]; then
    printf " │  Search: %-35s│\n" "${SEARXNG_URL} ✓"
else
    printf " │  Search: %-35s│\n" "UNAVAILABLE (watchdog retrying)"
fi
printf " │  Debug:  %-35s│\n" "http://localhost:8080/search/status"
echo " └─────────────────────────────────────────────┘"
echo ""
echo "  Open your browser to http://localhost:8080"
echo "  Press Ctrl+C to stop the server"
echo ""

# ── 7. Launch app ─────────────────────────────────────────────────────────────
cd ~/eliteomni_app 2>/dev/null || cd "$(dirname "$0")"

python -m uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1

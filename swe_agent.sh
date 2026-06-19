#!/bin/bash

TASK_DESCRIPTION=$1
CODEBASE_DIR=$2

if [ -z "$TASK_DESCRIPTION" ] || [ -z "$CODEBASE_DIR" ]; then
    echo "Usage: ./swe_agent.sh \"<task_description>\" \"/path/to/codebase\""
    exit 1
fi

cd "$CODEBASE_DIR" || exit 1

echo "[*] Initiating Long-Horizon Agent Loop..."
python3 "$(dirname "$0")/agent_core.py" "$TASK_DESCRIPTION"

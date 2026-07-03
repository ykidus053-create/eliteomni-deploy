#!/bin/bash
SRC="/home/kidus/eliteomni_app/app.py"
DST="/mnt/c/Users/kidus yared/Downloads/eliteomni/app.py"

echo "[Sync] Watching for changes..."
LAST_HASH=""

while true; do
    CURRENT_HASH=$(md5sum "$SRC" 2>/dev/null | cut -d' ' -f1)
    if [ "$CURRENT_HASH" != "$LAST_HASH" ] && [ -n "$CURRENT_HASH" ]; then
        cp "$SRC" "$DST"
        echo "[$(date '+%H:%M:%S')] Synced to Windows"
        LAST_HASH="$CURRENT_HASH"
    fi
    sleep 3
done

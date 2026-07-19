#!/usr/bin/env bash
# =============================================================================
# scripts/rclone_sync.sh
#
# Synchronizes the SQLite database (agent.db) with Yandex Disk or Google Drive.
#
# Supported modes:
#   rclone_sync.sh download   — downloads the database on node boot
#   rclone_sync.sh upload     — uploads the database before handover
# =============================================================================
set -euo pipefail

log() { echo -e "\033[1;32m[rclone-sync]\033[0m $*"; }
warn() { echo -e "\033[1;33m[rclone-sync][warn]\033[0m $*"; }

MODE="${1:-}"
if [[ "$MODE" != "download" && "$MODE" != "upload" && "$MODE" != "upload_signal" && "$MODE" != "check_signal" && "$MODE" != "delete_signal" ]]; then
    echo "Usage: $0 {download|upload|upload_signal|check_signal|delete_signal}"
    exit 1
fi

# Configuration via env vars (Kaggle secrets)
CLOUD_PROVIDER="${RCLONE_PROVIDER:-yadisk}" # yadisk | gdrive
REMOTE_NAME="backup"

# Configure Rclone via environment dynamically
if [[ "$CLOUD_PROVIDER" == "yadisk" || "$CLOUD_PROVIDER" == "yandex" ]]; then
    YANDEX_USER="${YANDEX_USER:-${RCLONE_USER:-}}"
    YANDEX_PASSWORD="${YANDEX_PASSWORD:-${RCLONE_PASS:-}}"
    YANDEX_TOKEN="${YANDEX_TOKEN:-${RCLONE_TOKEN:-}}"
    
    if [[ -n "$YANDEX_TOKEN" ]]; then
        # Yandex Disk native API config (Free Tier WebDAV bypass!)
        export RCLONE_CONFIG_BACKUP_TYPE=yandex
        export RCLONE_CONFIG_BACKUP_TOKEN="$YANDEX_TOKEN"
        REMOTE_PATH="backup:/sync"
    elif [[ -n "$YANDEX_USER" && -n "$YANDEX_PASSWORD" ]]; then
        # Yandex Disk WebDAV config (Requires paid Yandex 360)
        export RCLONE_CONFIG_BACKUP_TYPE=webdav
        export RCLONE_CONFIG_BACKUP_URL=https://webdav.yandex.ru
        export RCLONE_CONFIG_BACKUP_VENDOR=yandex
        export RCLONE_CONFIG_BACKUP_USER="$YANDEX_USER"
        export RCLONE_CONFIG_BACKUP_PASS=$(rclone obscure "$YANDEX_PASSWORD")
        REMOTE_PATH="backup:/sync"
    else
        warn "Yandex credentials (either YANDEX_TOKEN or YANDEX_USER/YANDEX_PASSWORD) are not set. Skipping sync."
        exit 0
    fi
    
elif [[ "$CLOUD_PROVIDER" == "gdrive" ]]; then
    GDRIVE_TOKEN="${GDRIVE_TOKEN:-}"
    
    if [[ -z "$GDRIVE_TOKEN" ]]; then
        warn "GDRIVE_TOKEN is not set. Skipping sync."
        exit 0
    fi
    
    # Google Drive config
    export RCLONE_CONFIG_BACKUP_TYPE=drive
    export RCLONE_CONFIG_BACKUP_CLIENT_ID="${GDRIVE_CLIENT_ID:-}"
    export RCLONE_CONFIG_BACKUP_CLIENT_SECRET="${GDRIVE_CLIENT_SECRET:-}"
    export RCLONE_CONFIG_BACKUP_TOKEN="$GDRIVE_TOKEN"
    REMOTE_PATH="backup:/kaggle_sync"
else
    warn "Unknown RCLONE_PROVIDER '$CLOUD_PROVIDER'. Skipping sync."
    exit 0
fi

DB_FILE="./data/agent.db"
DB_DIR=$(dirname "$DB_FILE")
mkdir -p "$DB_DIR"

if [[ "$MODE" == "download" ]]; then
    log "Downloading database from cloud storage..."
    # Check if remote folder or file exists
    if rclone lsf "$REMOTE_PATH" >/dev/null 2>&1; then
        if rclone copy "$REMOTE_PATH/agent.db" "$DB_DIR/" --progress; then
            log "Database successfully downloaded."
        else
            warn "No database backup found on remote directory. Starting fresh."
        fi
    else
        log "Remote folder does not exist yet. Starting with a fresh database."
    fi
elif [[ "$MODE" == "upload" ]]; then
    if [[ -f "$DB_FILE" ]]; then
        log "Uploading database to cloud storage..."
        rclone copy "$DB_FILE" "$REMOTE_PATH/" --progress
        log "Database successfully uploaded."
    else
        warn "Database file '$DB_FILE' not found. Nothing to upload."
    fi
elif [[ "$MODE" == "upload_signal" ]]; then
    log "Uploading handover signal..."
    mkdir -p logs
    echo "ready" > "logs/handover.signal"
    rclone copy "logs/handover.signal" "$REMOTE_PATH/"
    log "Handover signal successfully uploaded."
elif [[ "$MODE" == "check_signal" ]]; then
    if rclone lsf "$REMOTE_PATH/handover.signal" >/dev/null 2>&1; then
        exit 0
    else
        exit 1
    fi
elif [[ "$MODE" == "delete_signal" ]]; then
    log "Deleting handover signal from cloud..."
    rclone deletefile "$REMOTE_PATH/handover.signal" >/dev/null 2>&1 || true
    log "Handover signal deleted."
fi

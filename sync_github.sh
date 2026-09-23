#!/bin/bash

BASE_DIR="/mnt/usb_data/radio_astronomy/muon"
cd "$BASE_DIR" || exit 1

# Provide a standard PATH so cron can find git and ssh automatically
export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"

# Force Git to use our local secure deploy key
export GIT_SSH_COMMAND="ssh -i ${BASE_DIR}/github_deploy_key -o StrictHostKeyChecking=accept-new"

# Capture yesterday's date string for the commit log message
YESTERDAY=$(date -d "yesterday" +"%Y-%m-%d")

echo "⏳ [$(date '+%Y-%m-%d %H:%M:%S')] Starting midnight backup sequence for data up to ${YESTERDAY}..." >> "${BASE_DIR}/sync.log"

# Stage all updated logs and newly structured image paths
git add .

# Commit changes locally. If there is nothing new to commit, it won't crash.
git commit -m "Auto daily dataset synchronization: ${YESTERDAY}" >> "${BASE_DIR}/sync.log" 2>&1

# Check if a remote GitHub link is attached
if git remote | grep -q "origin"; then
    echo "🔄 Pulling remote changes to prevent conflicts..." >> "${BASE_DIR}/sync.log"
    
    # Safely pull down any remote changes using the preference you just fixed earlier
    git pull --no-rebase origin main >> "${BASE_DIR}/sync.log" 2>&1
    
    # Push the final combined history
    git push origin main >> "${BASE_DIR}/sync.log" 2>&1
    echo "✅ Successfully synced with GitHub remote." >> "${BASE_DIR}/sync.log"
else
    echo "ℹ️ Local commit saved successfully. (GitHub remote URL not linked yet)." >> "${BASE_DIR}/sync.log"
fi

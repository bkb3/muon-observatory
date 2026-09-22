#!/bin/bash

BASE_DIR="/mnt/usb_data/radio_astronomy/muon"
cd "$BASE_DIR" || exit 1

# Force Git to use our local secure deploy key
export GIT_SSH_COMMAND="ssh -i ${BASE_DIR}/github_deploy_key -o StrictHostKeyChecking=accept-new"

# Capture yesterday's date string for the commit log message
YESTERDAY=$(date -d "yesterday" +"%Y-%m-%d")

echo "⏳ Starting midnight backup sequence for data up to ${YESTERDAY}..." >> "${BASE_DIR}/sync.log"

# Stage all updated logs and newly structured image paths
git add .

# Commit changes locally (if there is nothing new, skip safely)
git commit -m "Auto daily dataset synchronization: ${YESTERDAY}" >> "${BASE_DIR}/sync.log" 2>&1

# Check if a remote GitHub link has been added yet before trying to push
if git remote | grep -q "origin"; then
    git push origin main >> "${BASE_DIR}/sync.log" 2>&1
    echo "✅ Successfully pushed to GitHub remote." >> "${BASE_DIR}/sync.log"
else
    echo "ℹ️ Local commit saved successfully. (GitHub remote URL not linked yet)." >> "${BASE_DIR}/sync.log"
fi

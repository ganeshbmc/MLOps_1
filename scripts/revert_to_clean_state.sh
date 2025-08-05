#!/bin/bash

set -e
set -x

# Optional: Save current uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "🔄 Stashing uncommitted changes before revert..."
  git stash push -u -m "Auto-stash before revert to pre-poisoned"
fi

# Revert back to clean state (Git tag)
git revert pre-poisoned..HEAD --no-commit
git commit -m "Reverted to pre-poisoned state"

# Restore clean data from DVC
dvc checkout --force
dvc pull
dvc status

# Optional: Restore stashed changes (uncomment if needed)
# git stash pop

# Show latest commits
git log --oneline --decorate -n 10

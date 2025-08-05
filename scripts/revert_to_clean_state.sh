#!/bin/bash

set -e
set -x

# Revert back to pre-poisoned state (assuming tag exists)
git revert pre-poisoned..HEAD --no-commit
git commit -m "Reverted to pre-poisoned state"

# Restore clean data from DVC
dvc checkout --force
dvc pull
dvc status

# Check current git location
git log --oneline --decorate -n 10
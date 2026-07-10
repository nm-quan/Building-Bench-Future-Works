#!/bin/bash
set -euo pipefail

# Only relevant for Claude Code on the web / remote sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Attribute commits made in this repo to the connected GitHub account
# (nm-quan) instead of the default "Claude" session identity, so commits
# link to the right GitHub profile/avatar.
git config user.name "KhoaCanTeam"
git config user.email "108200001+nm-quan@users.noreply.github.com"

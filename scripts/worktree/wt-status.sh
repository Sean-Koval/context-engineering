#!/bin/bash
# Get detailed status of a worktree
# Usage: wt-status.sh <type> <identifier>
#    or: wt-status.sh <path>

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Parse arguments
if [[ $# -eq 1 ]]; then
    # Direct path provided
    WORKTREE_PATH="$1"
elif [[ $# -eq 2 ]]; then
    TYPE="$1"
    IDENTIFIER="$2"
    WORKTREE_PATH=$(get_worktree_path "$TYPE" "$IDENTIFIER")
else
    echo "Usage: wt-status.sh <type> <identifier>"
    echo "   or: wt-status.sh <path>"
    exit 1
fi

if [[ ! -d "$WORKTREE_PATH" ]]; then
    echo "Error: Worktree not found at $WORKTREE_PATH"
    exit 1
fi

cd "$WORKTREE_PATH"

BRANCH=$(git branch --show-current)
COMMIT=$(git rev-parse --short HEAD)
COMMIT_MSG=$(git log -1 --format="%s")
COMMIT_DATE=$(git log -1 --format="%cr")

# Count changes
STAGED=$(git diff --cached --numstat | wc -l | tr -d ' ')
UNSTAGED=$(git diff --numstat | wc -l | tr -d ' ')
UNTRACKED=$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')

# Ahead/behind main
if git show-ref --verify --quiet refs/heads/main; then
    AHEAD_BEHIND=$(git rev-list --left-right --count main...HEAD 2>/dev/null || echo "0	0")
    BEHIND=$(echo "$AHEAD_BEHIND" | awk '{print $1}')
    AHEAD=$(echo "$AHEAD_BEHIND" | awk '{print $2}')
else
    BEHIND="?"
    AHEAD="?"
fi

# Recent commits on this branch
RECENT_COMMITS=$(git log main..HEAD --oneline 2>/dev/null | head -5)

echo "=== Worktree Status ==="
echo ""
echo "Path:        $WORKTREE_PATH"
echo "Branch:      $BRANCH"
echo "Commit:      $COMMIT ($COMMIT_DATE)"
echo "Message:     $COMMIT_MSG"
echo ""
echo "=== Changes ==="
echo "Staged:      $STAGED files"
echo "Unstaged:    $UNSTAGED files"
echo "Untracked:   $UNTRACKED files"
echo ""
echo "=== Vs Main ==="
echo "Ahead:       $AHEAD commits"
echo "Behind:      $BEHIND commits"
echo ""

if [[ -n "$RECENT_COMMITS" ]]; then
    echo "=== Commits on Branch ==="
    echo "$RECENT_COMMITS"
    echo ""
fi

# Show uncommitted changes summary
if [[ $((STAGED + UNSTAGED + UNTRACKED)) -gt 0 ]]; then
    echo "=== Uncommitted Files ==="
    git status --short
    echo ""
fi

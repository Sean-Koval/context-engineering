#!/bin/bash
# Sync a worktree with main branch
# Usage: wt-sync.sh <type> <identifier> [--rebase|--merge]
#    or: wt-sync.sh (from within a worktree)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Default strategy
STRATEGY="rebase"

# Parse arguments
WORKTREE_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebase)
            STRATEGY="rebase"
            shift
            ;;
        --merge)
            STRATEGY="merge"
            shift
            ;;
        *)
            if [[ -z "$TYPE" ]]; then
                TYPE="$1"
            elif [[ -z "$IDENTIFIER" ]]; then
                IDENTIFIER="$1"
            fi
            shift
            ;;
    esac
done

# Determine worktree path
if [[ -n "$TYPE" && -n "$IDENTIFIER" ]]; then
    WORKTREE_PATH=$(get_worktree_path "$TYPE" "$IDENTIFIER")
else
    # Try current directory
    WORKTREE_PATH=$(pwd)
fi

if [[ ! -d "$WORKTREE_PATH/.git" && ! -f "$WORKTREE_PATH/.git" ]]; then
    echo "Error: Not a git worktree: $WORKTREE_PATH"
    exit 1
fi

cd "$WORKTREE_PATH"

BRANCH=$(git branch --show-current)

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "Warning: You have uncommitted changes"
    echo ""
    git status --short
    echo ""
    read -p "Stash changes before syncing? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Aborting sync. Commit or stash your changes first."
        exit 1
    fi
    git stash push -m "Auto-stash before sync $(date -Iseconds)"
    STASHED=true
else
    STASHED=false
fi

# Fetch latest
echo "Fetching latest from origin..."
git fetch origin main:main 2>/dev/null || git fetch origin main || true

# Apply sync strategy
if [[ "$STRATEGY" == "rebase" ]]; then
    echo "Rebasing $BRANCH onto main..."
    if git rebase main; then
        echo "Rebase successful!"
    else
        echo "Rebase failed. Resolve conflicts and run: git rebase --continue"
        exit 1
    fi
else
    echo "Merging main into $BRANCH..."
    if git merge main -m "Merge main into $BRANCH"; then
        echo "Merge successful!"
    else
        echo "Merge failed. Resolve conflicts and commit."
        exit 1
    fi
fi

# Restore stashed changes
if $STASHED; then
    echo "Restoring stashed changes..."
    git stash pop
fi

echo ""
echo "=== Sync Complete ==="
echo "Branch $BRANCH is now up to date with main"

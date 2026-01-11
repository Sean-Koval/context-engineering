#!/bin/bash
# Remove a worktree and optionally its branch
# Usage: wt-remove.sh <type> <identifier> [--force] [--keep-branch]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Parse arguments
FORCE=false
KEEP_BRANCH=false
TYPE=""
IDENTIFIER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --keep-branch)
            KEEP_BRANCH=true
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

if [[ -z "$TYPE" || -z "$IDENTIFIER" ]]; then
    echo "Usage: wt-remove.sh <type> <identifier> [--force] [--keep-branch]"
    exit 1
fi

WORKTREE_PATH=$(get_worktree_path "$TYPE" "$IDENTIFIER")
BRANCH_NAME=$(get_branch_name "$TYPE" "$IDENTIFIER")

if [[ ! -d "$WORKTREE_PATH" ]]; then
    echo "Error: Worktree not found at $WORKTREE_PATH"
    exit 1
fi

# Check for uncommitted changes
cd "$WORKTREE_PATH"
if [[ -n $(git status --porcelain) ]]; then
    if $FORCE; then
        echo "Warning: Discarding uncommitted changes (--force)"
    else
        echo "Error: Worktree has uncommitted changes"
        git status --short
        echo ""
        echo "Use --force to discard changes, or commit them first."
        exit 1
    fi
fi

# Check if branch has unmerged commits
cd "$REPO_ROOT"
UNMERGED=$(git rev-list --count main.."$BRANCH_NAME" 2>/dev/null || echo "0")
if [[ "$UNMERGED" -gt 0 ]]; then
    if $FORCE; then
        echo "Warning: Discarding $UNMERGED unmerged commits (--force)"
    else
        echo "Warning: Branch has $UNMERGED unmerged commits"
        git log main.."$BRANCH_NAME" --oneline
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting. Use wt-finish.sh to create a PR first."
            exit 1
        fi
    fi
fi

# Remove worktree
echo "Removing worktree at $WORKTREE_PATH..."
if $FORCE; then
    git worktree remove --force "$WORKTREE_PATH"
else
    git worktree remove "$WORKTREE_PATH"
fi

# Remove branch unless --keep-branch
if ! $KEEP_BRANCH; then
    echo "Removing branch $BRANCH_NAME..."
    if $FORCE; then
        git branch -D "$BRANCH_NAME" 2>/dev/null || true
    else
        git branch -d "$BRANCH_NAME" 2>/dev/null || {
            echo "Branch not fully merged. Use --force to delete anyway."
        }
    fi
fi

# Update session registry
if [[ -f "$SESSION_REGISTRY" ]]; then
    jq --arg branch "$BRANCH_NAME" \
       --arg removed "$(date -Iseconds)" \
       'map(if .branch == $branch then . + {removed_at: $removed} else . end)' \
       "$SESSION_REGISTRY" > "${SESSION_REGISTRY}.tmp"
    mv "${SESSION_REGISTRY}.tmp" "$SESSION_REGISTRY"
fi

echo ""
echo "=== Worktree Removed ==="
echo "Path:   $WORKTREE_PATH (deleted)"
if ! $KEEP_BRANCH; then
    echo "Branch: $BRANCH_NAME (deleted)"
else
    echo "Branch: $BRANCH_NAME (kept)"
fi

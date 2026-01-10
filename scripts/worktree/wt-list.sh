#!/bin/bash
# List all worktrees with detailed status
# Usage: wt-list.sh [--json]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

JSON_OUTPUT=false
if [[ "$1" == "--json" ]]; then
    JSON_OUTPUT=true
fi

cd "$REPO_ROOT"

if $JSON_OUTPUT; then
    echo "["
    first=true
    while IFS= read -r worktree_line; do
        if [[ -z "$worktree_line" ]]; then continue; fi

        worktree_path=$(echo "$worktree_line" | awk '{print $1}')
        commit=$(echo "$worktree_line" | awk '{print $2}')
        branch=$(echo "$worktree_line" | sed 's/.*\[\(.*\)\].*/\1/' | tr -d '[]')

        # Get status
        if [[ -d "$worktree_path" ]]; then
            cd "$worktree_path"
            changes=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
            ahead_behind=$(git rev-list --left-right --count main...HEAD 2>/dev/null || echo "0	0")
            behind=$(echo "$ahead_behind" | awk '{print $1}')
            ahead=$(echo "$ahead_behind" | awk '{print $2}')
            cd "$REPO_ROOT"
        else
            changes=0
            behind=0
            ahead=0
        fi

        if ! $first; then echo ","; fi
        first=false

        cat <<EOF
  {
    "path": "$worktree_path",
    "branch": "$branch",
    "commit": "$commit",
    "uncommitted_changes": $changes,
    "ahead_of_main": $ahead,
    "behind_main": $behind
  }
EOF
    done < <(git worktree list)
    echo ""
    echo "]"
else
    echo "=== Git Worktrees ==="
    echo ""
    printf "%-50s %-25s %-8s %-10s\n" "PATH" "BRANCH" "CHANGES" "VS MAIN"
    printf "%-50s %-25s %-8s %-10s\n" "----" "------" "-------" "-------"

    while IFS= read -r worktree_line; do
        if [[ -z "$worktree_line" ]]; then continue; fi

        worktree_path=$(echo "$worktree_line" | awk '{print $1}')
        branch=$(echo "$worktree_line" | sed 's/.*\[\(.*\)\].*/\1/' | tr -d '[]')

        # Get status
        if [[ -d "$worktree_path" ]]; then
            cd "$worktree_path"
            changes=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
            ahead_behind=$(git rev-list --left-right --count main...HEAD 2>/dev/null || echo "0	0")
            behind=$(echo "$ahead_behind" | awk '{print $1}')
            ahead=$(echo "$ahead_behind" | awk '{print $2}')
            vs_main="+${ahead}/-${behind}"
            cd "$REPO_ROOT"
        else
            changes="?"
            vs_main="?"
        fi

        # Shorten path for display
        short_path="${worktree_path/#$WORKTREE_BASE\//}"

        printf "%-50s %-25s %-8s %-10s\n" "$short_path" "$branch" "$changes" "$vs_main"
    done < <(git worktree list)
    echo ""
fi

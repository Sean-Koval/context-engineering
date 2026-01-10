#!/bin/bash
# Finish a worktree session - prepare for PR or merge
# Usage: wt-finish.sh <type> <identifier> [--pr|--merge|--keep]
#
# Options:
#   --pr     Create a pull request (default)
#   --merge  Merge directly to main (for trusted sessions)
#   --keep   Keep worktree but mark session complete

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Parse arguments
ACTION="pr"
TYPE=""
IDENTIFIER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pr)
            ACTION="pr"
            shift
            ;;
        --merge)
            ACTION="merge"
            shift
            ;;
        --keep)
            ACTION="keep"
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
    echo "Usage: wt-finish.sh <type> <identifier> [--pr|--merge|--keep]"
    exit 1
fi

WORKTREE_PATH=$(get_worktree_path "$TYPE" "$IDENTIFIER")
BRANCH_NAME=$(get_branch_name "$TYPE" "$IDENTIFIER")

if [[ ! -d "$WORKTREE_PATH" ]]; then
    echo "Error: Worktree not found at $WORKTREE_PATH"
    exit 1
fi

cd "$WORKTREE_PATH"

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "Error: You have uncommitted changes in $WORKTREE_PATH"
    echo ""
    git status --short
    echo ""
    echo "Please commit or stash your changes first."
    exit 1
fi

# Get commit info for summary
COMMITS=$(git rev-list --count main..HEAD 2>/dev/null || echo "0")
FILES_CHANGED=$(git diff --stat main..HEAD 2>/dev/null | tail -1 || echo "0 files")

echo "=== Session Summary ==="
echo "Branch:   $BRANCH_NAME"
echo "Commits:  $COMMITS"
echo "Changes:  $FILES_CHANGED"
echo ""

case $ACTION in
    pr)
        echo "Creating pull request..."

        # Push branch to origin
        git push -u origin "$BRANCH_NAME" 2>/dev/null || git push origin "$BRANCH_NAME"

        # Create PR using gh if available
        if command -v gh &> /dev/null; then
            # Generate PR body
            PR_BODY=$(cat <<EOF
## Summary
Branch: \`$BRANCH_NAME\`
Commits: $COMMITS

## Changes
$(git log main..HEAD --oneline)

---
*Session: $TYPE/$IDENTIFIER*
EOF
)
            gh pr create --title "[$TYPE] $IDENTIFIER" --body "$PR_BODY" || {
                echo "PR creation failed. You can create it manually:"
                echo "  gh pr create --title '[$TYPE] $IDENTIFIER'"
            }
        else
            echo "gh CLI not found. Push complete. Create PR manually at:"
            echo "  https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/pull/new/$BRANCH_NAME"
        fi
        ;;

    merge)
        echo "Merging to main..."
        cd "$REPO_ROOT"
        git checkout main
        git merge "$BRANCH_NAME" --no-ff -m "Merge $BRANCH_NAME into main"
        echo "Merged successfully!"

        read -p "Remove worktree and delete branch? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            git worktree remove "$WORKTREE_PATH"
            git branch -d "$BRANCH_NAME"
            echo "Worktree and branch removed."
        fi
        ;;

    keep)
        echo "Session marked complete. Worktree preserved at:"
        echo "  $WORKTREE_PATH"
        echo ""
        echo "To remove later:"
        echo "  $SCRIPT_DIR/wt-remove.sh $TYPE $IDENTIFIER"
        ;;
esac

# Update session registry
if [[ -f "$SESSION_REGISTRY" ]]; then
    jq --arg branch "$BRANCH_NAME" \
       --arg action "$ACTION" \
       --arg finished "$(date -Iseconds)" \
       'map(if .branch == $branch then . + {finished_at: $finished, action: $action} else . end)' \
       "$SESSION_REGISTRY" > "${SESSION_REGISTRY}.tmp"
    mv "${SESSION_REGISTRY}.tmp" "$SESSION_REGISTRY"
fi

echo ""
echo "=== Session Finished ==="

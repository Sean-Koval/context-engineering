#!/bin/bash
# Create a new worktree with consistent naming
# Usage: wt-create.sh <type> <identifier> [--from <branch>] [--package <pkg>]
#
# Types: agent, feature, fix, experiment, review
#
# Examples:
#   wt-create.sh agent                           # Auto-numbered: agent/session-001
#   wt-create.sh feature compression-pipeline   # feature/compression-pipeline
#   wt-create.sh fix token-overflow --package context-core

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Parse arguments
TYPE=""
IDENTIFIER=""
FROM_BRANCH="main"
PACKAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            FROM_BRANCH="$2"
            shift 2
            ;;
        --package)
            PACKAGE="$2"
            shift 2
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

# Validate type
if [[ -z "$TYPE" ]]; then
    echo "Error: Type is required"
    echo "Valid types: ${VALID_TYPES[*]}"
    exit 1
fi

if ! validate_type "$TYPE"; then
    echo "Error: Invalid type '$TYPE'"
    echo "Valid types: ${VALID_TYPES[*]}"
    exit 1
fi

# Auto-generate identifier for agent type
if [[ "$TYPE" == "agent" && -z "$IDENTIFIER" ]]; then
    IDENTIFIER="session-$(get_next_agent_id)"
fi

if [[ -z "$IDENTIFIER" ]]; then
    echo "Error: Identifier is required for type '$TYPE'"
    exit 1
fi

# Build paths
WORKTREE_PATH=$(get_worktree_path "$TYPE" "$IDENTIFIER")
BRANCH_NAME=$(get_branch_name "$TYPE" "$IDENTIFIER")

# Check if worktree already exists
if [[ -d "$WORKTREE_PATH" ]]; then
    echo "Error: Worktree already exists at $WORKTREE_PATH"
    exit 1
fi

# Check if branch already exists
cd "$REPO_ROOT"
if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
    echo "Branch '$BRANCH_NAME' already exists. Using existing branch."
    git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"
else
    echo "Creating new branch '$BRANCH_NAME' from '$FROM_BRANCH'"
    git worktree add -b "$BRANCH_NAME" "$WORKTREE_PATH" "$FROM_BRANCH"
fi

# Update session registry
SESSION_INFO=$(cat <<EOF
{
  "type": "$TYPE",
  "identifier": "$IDENTIFIER",
  "branch": "$BRANCH_NAME",
  "path": "$WORKTREE_PATH",
  "package": "$PACKAGE",
  "created_at": "$(date -Iseconds)",
  "from_branch": "$FROM_BRANCH"
}
EOF
)

# Add to registry (create if doesn't exist)
if [[ -f "$SESSION_REGISTRY" ]]; then
    # Add to existing array
    jq --argjson new "$SESSION_INFO" '. += [$new]' "$SESSION_REGISTRY" > "${SESSION_REGISTRY}.tmp"
    mv "${SESSION_REGISTRY}.tmp" "$SESSION_REGISTRY"
else
    echo "[$SESSION_INFO]" | jq '.' > "$SESSION_REGISTRY"
fi

echo ""
echo "=== Worktree Created ==="
echo "Path:     $WORKTREE_PATH"
echo "Branch:   $BRANCH_NAME"
echo "From:     $FROM_BRANCH"
if [[ -n "$PACKAGE" ]]; then
    echo "Package:  $PACKAGE"
fi
echo ""
echo "To start working:"
echo "  cd $WORKTREE_PATH"
echo "  claude"
echo ""
echo "When finished:"
echo "  $SCRIPT_DIR/wt-finish.sh $TYPE $IDENTIFIER"

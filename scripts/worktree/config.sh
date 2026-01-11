#!/bin/bash
# Worktree Management Configuration
# Used by all worktree scripts for consistent naming and paths

# Base directory for worktrees (sibling to main repo)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
WORKTREE_BASE="$(dirname "$REPO_ROOT")"

# Naming convention: {repo}-{type}-{identifier}
# Types: agent, feature, fix, experiment, review
VALID_TYPES=("agent" "feature" "fix" "experiment" "review")

# Session registry file (tracks active agent sessions)
SESSION_REGISTRY="$REPO_ROOT/.worktree-sessions.json"

# Branch naming: {type}/{identifier}
# Examples:
#   agent/session-001
#   feature/compression-pipeline
#   fix/token-budget-overflow

# Worktree directory naming: {repo}-{type}-{identifier}
# Examples:
#   context-engineering-agent-001
#   context-engineering-feature-compression

get_worktree_path() {
    local type="$1"
    local identifier="$2"
    echo "${WORKTREE_BASE}/${REPO_NAME}-${type}-${identifier}"
}

get_branch_name() {
    local type="$1"
    local identifier="$2"
    echo "${type}/${identifier}"
}

validate_type() {
    local type="$1"
    for valid in "${VALID_TYPES[@]}"; do
        if [[ "$type" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# Generate next agent session number
get_next_agent_id() {
    local max_id=0
    while IFS= read -r line; do
        if [[ "$line" =~ agent-([0-9]+) ]]; then
            id="${BASH_REMATCH[1]}"
            if (( id > max_id )); then
                max_id=$id
            fi
        fi
    done < <(git worktree list --porcelain | grep "^worktree" | sed 's/worktree //')
    echo "$(printf "%03d" $((max_id + 1)))"
}

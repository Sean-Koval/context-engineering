#!/bin/bash
# Git Worktree Manager for Agent Sessions
# Usage: wt <command> [args...]
#
# Commands:
#   list                           List all worktrees with status
#   create <type> <id> [options]   Create a new worktree
#   status <type> <id>             Show worktree status
#   sync <type> <id> [options]     Sync with main branch
#   finish <type> <id> [options]   Finish session (create PR)
#   remove <type> <id> [options]   Remove worktree
#   sessions                       Show active sessions from registry
#   help                           Show this help
#
# Types: agent, feature, fix, experiment, review
#
# Examples:
#   wt create agent                     # Create agent/session-001
#   wt create feature compression       # Create feature/compression
#   wt list                             # List all worktrees
#   wt status agent 001                 # Show status of agent-001
#   wt finish feature compression --pr  # Create PR for feature
#   wt remove agent 001                 # Remove completed worktree

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

show_help() {
    cat <<'EOF'
Git Worktree Manager for Agent Sessions

USAGE:
    wt <command> [arguments...]

COMMANDS:
    list                              List all worktrees with status
    create <type> [id] [options]      Create a new worktree
    status <type> <id>                Show detailed worktree status
    sync [type] [id] [options]        Sync worktree with main
    finish <type> <id> [options]      Finish session (create PR)
    remove <type> <id> [options]      Remove worktree
    sessions                          Show session registry
    help                              Show this help

TYPES:
    agent       Auto-numbered agent sessions (agent/session-001)
    feature     Feature development (feature/my-feature)
    fix         Bug fixes (fix/issue-123)
    experiment  Experimental work (experiment/try-thing)
    review      Code review sessions (review/pr-456)

CREATE OPTIONS:
    --from <branch>     Base branch (default: main)
    --package <pkg>     Associate with a package

SYNC OPTIONS:
    --rebase            Rebase onto main (default)
    --merge             Merge main into branch

FINISH OPTIONS:
    --pr                Create pull request (default)
    --merge             Merge directly to main
    --keep              Keep worktree, mark complete

REMOVE OPTIONS:
    --force, -f         Force removal even with changes
    --keep-branch       Don't delete the branch

EXAMPLES:
    # Start a new agent session
    wt create agent
    cd ../context-engineering-agent-session-001
    claude

    # Create feature branch for compression work
    wt create feature compression-pipeline --package context-compression

    # Check status of all worktrees
    wt list

    # Sync with latest main
    wt sync feature compression-pipeline

    # Finish and create PR
    wt finish feature compression-pipeline

    # Clean up
    wt remove feature compression-pipeline
EOF
}

show_sessions() {
    if [[ ! -f "$SESSION_REGISTRY" ]]; then
        echo "No sessions registered yet."
        exit 0
    fi

    echo "=== Session Registry ==="
    echo ""
    jq -r '.[] | "[\(.type)] \(.identifier)\n  Branch:  \(.branch)\n  Created: \(.created_at)\n  Package: \(.package // "none")\n  Status:  \(if .removed_at then "removed" elif .finished_at then "finished (\(.action))" else "active" end)\n"' "$SESSION_REGISTRY"
}

# Main command dispatcher
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    list|ls)
        "$SCRIPT_DIR/wt-list.sh" "$@"
        ;;
    create|new)
        "$SCRIPT_DIR/wt-create.sh" "$@"
        ;;
    status|st)
        "$SCRIPT_DIR/wt-status.sh" "$@"
        ;;
    sync)
        "$SCRIPT_DIR/wt-sync.sh" "$@"
        ;;
    finish|done)
        "$SCRIPT_DIR/wt-finish.sh" "$@"
        ;;
    remove|rm)
        "$SCRIPT_DIR/wt-remove.sh" "$@"
        ;;
    sessions)
        show_sessions
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run 'wt help' for usage."
        exit 1
        ;;
esac

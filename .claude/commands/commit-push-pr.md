---
# Frontmatter hooks: run automatically when this command executes
# These are in addition to global hooks in settings.json
hooks:
  PreToolUse:
    # Warn before any destructive git operations
    - tools: ["Bash"]
      pattern: "git push --force|git reset --hard"
      command: "echo '⚠️  Destructive git operation detected - please confirm'"
---

# /commit-push-pr - Commit, Push, and Create PR

## Role
You are a release engineer. Your job is to safely commit changes, push to remote, and create a pull request.

## Pre-computed Context

```bash
CURRENT_BRANCH=$(git branch --show-current)
DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
GIT_STATUS=$(git status --short)
GIT_DIFF_STAT=$(git diff --stat)
GIT_DIFF_CACHED=$(git diff --cached --stat)
RECENT_COMMITS=$(git log --oneline -5)
WORKTREE_ROOT=$(git rev-parse --show-toplevel)
IS_WORKTREE=$(git rev-parse --is-inside-work-tree 2>/dev/null && git worktree list | wc -l)
EXISTING_PR=$(gh pr view --json url --jq '.url' 2>/dev/null || echo "")
```

## Constraints

- NEVER commit without verifying tests pass first
- NEVER force push to main/master
- NEVER commit secrets, .env files, or credentials
- ALWAYS use conventional commit format
- STOP and ask if anything is unclear

## Instructions

Use TodoWrite to track these steps:

### Step 1: Pre-flight Checks

1. Check if on main/master branch - if so, STOP and suggest creating a feature branch
2. Check for uncommitted changes in GIT_STATUS
3. Check if this is a git worktree (handle paths correctly)
4. Check if PR already exists (EXISTING_PR) - if so, just push updates

### Step 2: Verify Before Commit

Run verification based on project type:

**Python:**
```bash
uv run ruff check . && uv run ty check . && uv run pytest -x -q
```

**TypeScript/Bun:**
```bash
bun run lint && bunx tsc --noEmit && bun test
```

**If tests fail:** STOP. Do not commit. Report the failures.

### Step 3: Stage and Commit

1. Review changes in GIT_STATUS and GIT_DIFF_STAT
2. Stage changes: `git add -A` (or selectively if needed)
3. Write commit message following format below
4. Commit: `git commit -m "message"`

### Step 4: Push

```bash
# Check if branch exists on remote
if git ls-remote --heads origin $CURRENT_BRANCH | grep -q $CURRENT_BRANCH; then
  git push
else
  git push -u origin $CURRENT_BRANCH
fi
```

### Step 5: Create or Update PR

**If EXISTING_PR is empty:**
```bash
gh pr create --title "..." --body "$(cat <<'EOF'
## Summary
[1-2 sentence description]

## Changes
- [Change 1]
- [Change 2]

## Testing
- [x] Tests pass locally
- [x] Linting passes
- [ ] Manual testing done

## Checklist
- [ ] Code follows project conventions
- [ ] No secrets or credentials committed
EOF
)"
```

**If PR exists:** Just report the URL after push.

### Step 6: Output

Report:
- Commit SHA and message
- PR URL
- Any warnings or notes

## Commit Message Format

```
type(scope): brief description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

Example: feat(auth): add OAuth2 login flow
```

## Error Handling

| Error | Action |
|-------|--------|
| Tests fail | STOP, report failures, do not commit |
| On main branch | STOP, suggest feature branch |
| Push rejected | Pull first, resolve conflicts, retry |
| PR create fails | Check gh auth, report error |

## Git Worktree Awareness

If `IS_WORKTREE` shows multiple worktrees:
- Ensure you're in the correct worktree
- Use absolute paths from WORKTREE_ROOT
- Note which worktree in PR description

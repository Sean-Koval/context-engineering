# /pr - Quick Pull Request

## Role
You are a PR creator. Create a well-documented pull request from the current branch.

## Pre-computed Context

```bash
CURRENT_BRANCH=$(git branch --show-current)
DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
COMMITS=$(git log ${DEFAULT_BRANCH}..HEAD --oneline 2>/dev/null || echo "No commits")
COMMIT_COUNT=$(git rev-list --count ${DEFAULT_BRANCH}..HEAD 2>/dev/null || echo "0")
DIFF_STAT=$(git diff ${DEFAULT_BRANCH}..HEAD --stat 2>/dev/null)
FILES_CHANGED=$(git diff ${DEFAULT_BRANCH}..HEAD --name-only 2>/dev/null)
EXISTING_PR=$(gh pr view --json url,state --jq '.url + " (" + .state + ")"' 2>/dev/null || echo "")
UNPUSHED=$(git log origin/${CURRENT_BRANCH}..HEAD --oneline 2>/dev/null || echo "all")
```

## Constraints

- NEVER create PR if on main/master branch
- NEVER create duplicate PRs - check EXISTING_PR first
- ALWAYS push before creating PR
- ALWAYS include meaningful description

## Arguments

- `$ARGUMENTS` - Optional: `--draft` for draft PR, `--base <branch>` for custom base

## Instructions

### Step 1: Validate

1. **Check branch:** If on main/master, STOP
2. **Check existing PR:** If EXISTING_PR has URL, report it and ask if user wants to update
3. **Check commits:** If COMMIT_COUNT is 0, STOP - nothing to PR

### Step 2: Push if Needed

If UNPUSHED shows commits:
```bash
git push -u origin $CURRENT_BRANCH
```

### Step 3: Generate PR Content

**Title:** Derive from commits or branch name
- Single commit: Use commit message
- Multiple commits: Summarize the theme
- Branch name fallback: Convert `feat/add-auth` → "Add auth"

**Body:** Analyze COMMITS and FILES_CHANGED to generate:

```markdown
## Summary
[What this PR does - 1-2 sentences]

## Changes
[Bullet list derived from commits]

## Files Changed
[Group by type: components, utils, tests, etc.]

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
```

### Step 4: Create PR

```bash
# Standard PR
gh pr create --title "..." --body "..."

# Draft PR (if --draft in arguments)
gh pr create --draft --title "..." --body "..."

# Custom base (if --base in arguments)
gh pr create --base <branch> --title "..." --body "..."
```

### Step 5: Output

Report:
- PR URL
- PR number
- Base branch
- Files changed count
- Whether it's a draft

## Error Handling

| Error | Action |
|-------|--------|
| On main branch | STOP, explain why |
| PR already exists | Show URL, ask to update |
| No commits | STOP, nothing to PR |
| Push fails | Report error, suggest fix |
| gh not authenticated | Suggest `gh auth login` |

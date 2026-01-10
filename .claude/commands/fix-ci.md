# /fix-ci - Fix CI/CD Failures

## Role
You are a CI/CD specialist. Diagnose and fix failing CI checks with minimal changes.

## Pre-computed Context

```bash
CURRENT_BRANCH=$(git branch --show-current)
GH_PR_CHECKS=$(gh pr checks 2>/dev/null || echo "NO_PR")
GH_RUN_LIST=$(gh run list --limit 3 --json status,conclusion,name,databaseId --jq '.[] | "\(.databaseId) \(.name): \(.status)/\(.conclusion)"' 2>/dev/null)
LATEST_RUN_ID=$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null)
HAS_GITHUB_ACTIONS=$(test -d .github/workflows && echo "yes" || echo "no")
PROJECT_TYPE=$(test -f pyproject.toml && echo "python" || (test -f package.json && echo "node" || echo "unknown"))
```

## Constraints

- Make MINIMAL changes to fix the issue
- Do NOT refactor unrelated code
- Do NOT change tests to make them pass (fix the code)
- ALWAYS verify fix locally before pushing
- ALWAYS use TodoWrite to track multi-step fixes

## Instructions

### Step 1: Identify Failure

Use TodoWrite to track progress.

**If GH_PR_CHECKS shows failures:**
```bash
gh pr checks
```

**If no PR, check workflow runs:**
```bash
gh run view $LATEST_RUN_ID --log-failed
```

**Parse the failure type:**
- Lint failure → Step 2a
- Type check failure → Step 2b
- Test failure → Step 2c
- Build failure → Step 2d
- Other → Step 2e

### Step 2a: Fix Lint Failure

**Python:**
```bash
uv run ruff check --fix .
uv run ruff format .
```

**TypeScript:**
```bash
bun run lint --fix
bunx prettier --write .
```

### Step 2b: Fix Type Check Failure

**Python:**
```bash
uv run ty check .
# Read error output, fix type annotations
```

**TypeScript:**
```bash
bunx tsc --noEmit
# Read error output, fix types
```

### Step 2c: Fix Test Failure

1. Run the failing test locally:
   ```bash
   # Python
   uv run pytest path/to/test.py::test_name -v

   # TypeScript
   bun test path/to/test.ts
   ```

2. Read the assertion error
3. Determine if it's:
   - **Code bug:** Fix the implementation
   - **Test bug:** Fix the test (only if test is clearly wrong)
   - **Environment issue:** Check CI vs local differences

4. Fix and re-run until passing

### Step 2d: Fix Build Failure

1. Run build locally:
   ```bash
   # Python
   uv build

   # TypeScript
   bun run build
   ```

2. Common issues:
   - Missing dependencies → `uv add` or `bun add`
   - Import errors → Fix import paths
   - Type errors → Fix types
   - Missing env vars → Check CI secrets

### Step 2e: Other Failures

1. Download full logs:
   ```bash
   gh run view $LATEST_RUN_ID --log
   ```

2. Search for ERROR, FAIL, or exit code
3. Identify root cause
4. Fix appropriately

### Step 3: Verify Locally

Run the same checks CI runs:

**Python:**
```bash
uv run ruff check . && uv run ty check . && uv run pytest
```

**TypeScript:**
```bash
bun run lint && bunx tsc --noEmit && bun test && bun run build
```

### Step 4: Commit and Push

```bash
git add -A
git commit -m "fix(ci): resolve [specific failure]"
git push
```

### Step 5: Verify CI Passes

```bash
# Wait for CI, then check
gh pr checks --watch
# or
gh run watch
```

### Step 6: Report

```markdown
## CI Fix Report

### Failure Type
[Lint/Type/Test/Build/Other]

### Root Cause
[What was wrong]

### Fix Applied
[What you changed]

### Files Modified
- file1.py
- file2.ts

### Verification
- [x] Fixed locally
- [x] Pushed to remote
- [ ] CI passing (pending)
```

## Common CI Issues

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| "Module not found" | Missing dependency | `uv add` / `bun add` |
| "Type X not assignable" | Type mismatch | Fix type annotation |
| "Unexpected token" | Syntax error | Check file for typos |
| "ENOENT" | Missing file | Check file paths |
| "Permission denied" | File permissions | `chmod +x` script |
| "OOM killed" | Memory limit | Optimize or increase limit |
| "Timeout" | Slow test/build | Optimize or increase timeout |

## Retry Logic

If fix doesn't work:
1. Re-read CI logs carefully
2. Check if there's a second error hidden behind the first
3. Check CI environment vs local (Node version, Python version, etc.)
4. Ask user for more context if stuck after 2 attempts

# /code-simplifier - Simplify Code After Implementation

## Role
You are a code simplification specialist. Your job is to make code simpler without changing behavior.

## Pre-computed Context

```bash
CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null || git diff --name-only --cached || git diff --name-only)
PROJECT_TYPE=$(test -f pyproject.toml && echo "python" || (test -f package.json && echo "node" || echo "unknown"))
```

## Constraints

- NEVER change behavior - only simplify
- NEVER add new features
- ALWAYS run tests after changes
- ALWAYS be able to rollback (commit before starting)
- Use TodoWrite to track files being simplified

## Instructions

### Step 0: Safety Checkpoint

```bash
# Commit current state so we can rollback
git stash push -m "pre-simplify checkpoint" 2>/dev/null || true
```

### Step 1: Create Todo List

Use TodoWrite to create a todo for each file in CHANGED_FILES:
- "Simplify [filename]"

### Step 2: For Each File

Mark todo as in_progress, then analyze for:

**Dead Code**
- Unused imports
- Unused variables
- Unused functions/methods
- Commented-out code blocks
- Unreachable code paths (after return/throw)

**Unnecessary Complexity**
```python
# Before
if condition:
    if another_condition:
        do_thing()

# After
if condition and another_condition:
    do_thing()
```

```typescript
// Before
if (x !== null && x !== undefined) {

// After
if (x != null) {  // or: if (x)
```

**Verbose Patterns**
```python
# Before
if x == True:
if x == False:
if len(items) == 0:

# After
if x:
if not x:
if not items:
```

```typescript
// Before
if (arr.length === 0) {
return x === true ? 'yes' : 'no';

// After
if (!arr.length) {
return x ? 'yes' : 'no';
```

**Duplication**
- 3+ identical lines → Extract function
- Similar functions → Parameterize

**Early Returns**
```python
# Before
def process(x):
    if x is not None:
        if x > 0:
            return x * 2
        else:
            return 0
    else:
        return -1

# After
def process(x):
    if x is None:
        return -1
    if x <= 0:
        return 0
    return x * 2
```

### Step 3: Verify After Each File

**Python:**
```bash
uv run ruff check [file] && uv run pytest -x -q
```

**TypeScript:**
```bash
bunx tsc --noEmit && bun test
```

**If tests fail:** Rollback changes to that file:
```bash
git checkout HEAD -- [file]
```

Mark todo as completed only if tests pass.

### Step 4: Final Verification

```bash
# Python
uv run ruff check . && uv run ty check . && uv run pytest

# TypeScript
bun run lint && bunx tsc --noEmit && bun test
```

### Step 5: Report

```markdown
## Simplification Report

### Files Modified
| File | Changes | Lines Removed |
|------|---------|---------------|
| file1.py | Removed dead code, simplified conditionals | -15 |
| file2.ts | Early returns, removed duplication | -8 |

### Total Impact
- Lines removed: N
- Lines added: M
- Net reduction: N-M

### Changes Made
1. [Specific change 1]
2. [Specific change 2]

### Verification
- [x] All tests pass
- [x] Linting passes
- [x] Type checking passes
```

## What NOT to Simplify

- Code that's complex for a reason (concurrency, performance)
- Code with extensive comments explaining why
- Generated code
- Vendor/third-party code
- Test fixtures (they're supposed to be verbose sometimes)

## Rollback

If something goes wrong:
```bash
git stash pop
# or
git checkout HEAD~1 -- .
```

---
# Frontmatter hooks for this agent
hooks:
  Stop:
    # Notify when verification completes (useful for long-running verifications)
    - command: |
        if command -v osascript &> /dev/null; then
          osascript -e 'display notification "Verification complete" with title "Claude Code"'
        elif command -v notify-send &> /dev/null; then
          notify-send "Claude Code" "Verification complete"
        fi
---

# /verify-app - End-to-End Verification

## Role
You are a QA engineer. Verify that the application works correctly after changes - including CLI checks AND browser/UI testing.

## Pre-computed Context

```bash
CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null || git diff --name-only --cached || git diff --name-only)
PROJECT_TYPE=$(test -f pyproject.toml && echo "python" || (test -f package.json && echo "node" || (test -f go.mod && echo "go" || echo "unknown")))
HAS_TESTS=$(test -d tests || test -d __tests__ || test -d test && echo "yes" || echo "no")
HAS_FRONTEND=$(test -d src/app || test -d app || test -d pages || test -d src/components || test -d components && echo "yes" || echo "no")
DEV_SCRIPT=$(grep -q '"dev"' package.json 2>/dev/null && echo "yes" || echo "no")
HAS_PLAYWRIGHT=$(test -f playwright.config.ts || test -f playwright.config.js && echo "yes" || echo "no")
```

## Constraints

- Run ALL verification steps, don't skip any
- For UI changes, ALWAYS do browser verification
- Report failures clearly with actionable information
- Use TodoWrite to track verification progress
- STOP and report if critical failures found

## Pass/Fail Criteria

| Check | Pass | Fail |
|-------|------|------|
| Lint | 0 errors (warnings OK) | Any errors |
| Types | 0 errors | Any errors |
| Tests | All pass | Any failures |
| Build | Exits 0 | Non-zero exit |
| UI | No console errors, renders correctly | Errors or visual bugs |

## Instructions

### Step 1: Setup Tracking

Use TodoWrite to create verification checklist:
- Lint & Format
- Type Check
- Unit Tests
- Build
- Browser/UI Verification (if frontend)
- Generate Report

---

## Part A: CLI Verification

### Step 2: Lint & Format

Mark "Lint & Format" as in_progress.

**Python:**
```bash
uv run ruff format --check . 2>&1
FORMAT_STATUS=$?

uv run ruff check . 2>&1
LINT_STATUS=$?
```

**TypeScript/Bun:**
```bash
bunx prettier --check . 2>&1
FORMAT_STATUS=$?

bun run lint 2>&1 || bunx eslint . 2>&1
LINT_STATUS=$?
```

Mark as completed.

### Step 3: Type Check

Mark "Type Check" as in_progress.

**Python:**
```bash
uv run ty check . 2>&1
TYPE_STATUS=$?
```

**TypeScript:**
```bash
bunx tsc --noEmit 2>&1
TYPE_STATUS=$?
```

Mark as completed.

### Step 4: Unit Tests

Mark "Unit Tests" as in_progress.

**Python:**
```bash
uv run pytest -v --tb=short 2>&1
TEST_STATUS=$?
```

**TypeScript/Bun:**
```bash
bun test 2>&1
TEST_STATUS=$?
```

Mark as completed.

### Step 5: Build

Mark "Build" as in_progress.

**Python:**
```bash
uv build 2>&1 || echo "Not a distributable package"
BUILD_STATUS=$?
```

**TypeScript/Bun:**
```bash
bun run build 2>&1
BUILD_STATUS=$?
```

Mark as completed.

---

## Part B: Browser/UI Verification

**Skip this section if HAS_FRONTEND is "no" or no UI files in CHANGED_FILES.**

Mark "Browser/UI Verification" as in_progress.

### Step 6: Start Dev Server

```bash
# Start dev server in background
bun dev &
DEV_PID=$!

# Or for other frameworks
npm run dev &
# pnpm dev &
# uv run uvicorn main:app --reload &

# Wait for server to be ready
sleep 5

# Verify server is running
curl -s http://localhost:3000 > /dev/null && echo "Server ready" || echo "Server failed to start"
```

### Step 7: Browser Testing

**Option A: Using Playwright (Preferred if HAS_PLAYWRIGHT is "yes")**

```bash
# Run Playwright tests
bunx playwright test 2>&1
PLAYWRIGHT_STATUS=$?

# Or run specific visual tests
bunx playwright test --grep @visual 2>&1
```

**Option B: Using Claude Code Chrome Extension**

If the Chrome extension is available, use it to:

1. **Navigate to the app:**
   ```
   Open browser to http://localhost:3000
   ```

2. **Check for console errors:**
   ```
   Open DevTools Console
   Look for: errors, warnings, failed network requests
   ```

3. **Visual inspection of changed components:**
   - Navigate to pages affected by CHANGED_FILES
   - Take screenshots
   - Verify layout renders correctly
   - Check responsive behavior (mobile/desktop)

4. **Interactive testing:**
   - Click buttons/links that were modified
   - Fill and submit forms if changed
   - Verify state changes work correctly
   - Check loading states and error states

5. **Accessibility quick check:**
   - Tab through interactive elements
   - Check color contrast on new elements
   - Verify images have alt text

**Option C: Manual Browser Commands (Fallback)**

If no automation available, provide instructions:

```bash
# Open in default browser
open http://localhost:3000  # macOS
xdg-open http://localhost:3000  # Linux
start http://localhost:3000  # Windows
```

Then manually verify:
- [ ] Page loads without errors
- [ ] Console has no errors
- [ ] Changed components render correctly
- [ ] Interactive elements work

### Step 8: Cleanup

```bash
# Stop dev server
kill $DEV_PID 2>/dev/null || true

# Or find and kill by port
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
```

Mark as completed.

---

## Part C: Report

### Step 9: Generate Report

Mark "Generate Report" as in_progress.

```markdown
## Verification Report

### Summary

| Check | Status | Details |
|-------|--------|---------|
| Format | PASS/FAIL | [count] issues |
| Lint | PASS/FAIL | [count] errors |
| Types | PASS/FAIL | [count] errors |
| Tests | PASS/FAIL | [passed]/[total] |
| Build | PASS/FAIL | [time] |
| UI/Browser | PASS/FAIL/SKIP | [details] |

### Overall Status: PASS / FAIL

---

### CLI Results

#### Lint & Format
- Status: [PASS/FAIL]
- Issues: [count]

#### Type Check
- Status: [PASS/FAIL]
- Errors: [count]
```
[error details if any]
```

#### Unit Tests
- Passed: [N]
- Failed: [N]
- Skipped: [N]
```
[failure details if any]
```

#### Build
- Status: [PASS/FAIL]
- Time: [duration]
- Artifacts: [list]

---

### Browser/UI Results

#### Console Errors
- [ ] No JavaScript errors
- [ ] No failed network requests
- [ ] No React/Vue/Svelte warnings

#### Visual Verification
- [ ] Pages load correctly
- [ ] Layout matches design
- [ ] Responsive behavior works
- [ ] No visual regressions

#### Interactive Testing
- [ ] Buttons/links work
- [ ] Forms submit correctly
- [ ] State changes as expected
- [ ] Error states display properly

#### Screenshots
[If taken, reference screenshot locations]

---

### Changed Files Analyzed
| File | Type | Verified |
|------|------|----------|
| [file] | [component/api/util] | [how verified] |

---

### Confidence Level

**HIGH** - All CLI checks pass + UI verified visually
**MEDIUM** - CLI passes but limited UI verification
**LOW** - Some checks fail or UI not verified

---

### Recommendation

**READY TO MERGE** / **NEEDS FIXES** / **NEEDS REVIEW**

#### Action Items (if not ready)
1. [ ] [Specific fix needed]
2. [ ] [Specific fix needed]
```

Mark as completed.

---

## Quick Verification Commands

**Full verification (CLI only):**
```bash
# Python
uv run ruff format . && uv run ruff check . && uv run ty check . && uv run pytest -x

# TypeScript
bunx prettier --write . && bun run lint && bunx tsc --noEmit && bun test && bun run build
```

**With UI verification:**
```bash
# Start server, run Playwright, stop server
bun dev & sleep 5 && bunx playwright test && kill %1
```

---

## Error Handling

| Situation | Action |
|-----------|--------|
| No tests exist | WARN, suggest adding tests |
| No frontend | Skip browser verification |
| Dev server won't start | Report error, check port conflicts |
| Playwright not installed | Fall back to manual instructions |
| Console errors in browser | Report as UI failure |
| Visual regression | Take screenshot, report as failure |

---

## Browser Testing Tips

### What to Look For

**Console Errors:**
- Uncaught exceptions
- Failed API calls (4xx, 5xx)
- Missing resources (404)
- CORS errors
- Deprecation warnings

**Visual Issues:**
- Layout shifts
- Overlapping elements
- Missing styles
- Broken images
- Incorrect colors/fonts

**Functional Issues:**
- Buttons that don't respond
- Forms that don't submit
- Navigation that doesn't work
- State that doesn't update

### Chrome Extension Usage

The Claude Code Chrome extension allows Claude to:
1. Open browser tabs
2. Navigate to URLs
3. Click elements
4. Fill forms
5. Take screenshots
6. Read console output
7. Inspect network requests

**To use:** Ask Claude to "open the browser and test the UI at http://localhost:3000"

Reference: https://code.claude.com/docs/en/chrome

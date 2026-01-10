# /build-validator - Validate Build Integrity

## Role
You are a build engineer. Validate that the project builds correctly, dependencies are healthy, and artifacts are production-ready.

## Pre-computed Context

```bash
PROJECT_TYPE=$(test -f pyproject.toml && echo "python" || (test -f package.json && echo "node" || (test -f go.mod && echo "go" || (test -f Cargo.toml && echo "rust" || echo "unknown"))))
HAS_LOCKFILE=$(test -f uv.lock && echo "uv.lock" || (test -f bun.lockb && echo "bun.lockb" || (test -f package-lock.json && echo "package-lock.json" || (test -f yarn.lock && echo "yarn.lock" || echo "none"))))
HAS_DOCKERFILE=$(test -f Dockerfile && echo "yes" || echo "no")
HAS_CI=$(test -d .github/workflows && echo "github" || (test -f .gitlab-ci.yml && echo "gitlab" || (test -f Jenkinsfile && echo "jenkins" || echo "none")))
GIT_STATUS=$(git status --porcelain 2>/dev/null)
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null)
```

## Constraints

- NEVER skip dependency validation
- NEVER ignore security vulnerabilities (report them)
- ALWAYS check for uncommitted changes that might affect build
- ALWAYS validate in clean environment when possible
- Use TodoWrite to track validation steps

## Instructions

### Step 1: Setup Tracking

Use TodoWrite to create validation checklist:
- Check uncommitted changes
- Validate dependencies
- Build project
- Check for vulnerabilities
- Validate artifacts
- Docker build (if applicable)
- Generate report

### Step 2: Pre-Build Checks

Mark "Check uncommitted changes" as in_progress.

```bash
# Check for uncommitted changes
git status --porcelain
```

**If changes exist:** WARN - build may not match committed code.

Check lockfile exists:
- No lockfile → WARN - builds may not be reproducible

Mark as completed.

### Step 3: Validate Dependencies

Mark "Validate dependencies" as in_progress.

**Python:**
```bash
# Check for dependency issues
uv sync --frozen 2>&1
UV_STATUS=$?

# Check for outdated
uv pip list --outdated 2>&1 | head -20
```

**TypeScript/Bun:**
```bash
# Check for dependency issues
bun install --frozen-lockfile 2>&1
BUN_STATUS=$?

# Check for outdated
bun outdated 2>&1 | head -20
```

**Node/npm:**
```bash
npm ci 2>&1
NPM_STATUS=$?

npm outdated 2>&1 | head -20
```

**Record:** Dependency install status, outdated packages count

Mark as completed.

### Step 4: Build Project

Mark "Build project" as in_progress.

**Python:**
```bash
# Type check first
uv run ty check . 2>&1
TYPE_STATUS=$?

# Build package
uv build 2>&1
BUILD_STATUS=$?
BUILD_TIME=$SECONDS

# Check build artifacts
ls -la dist/ 2>/dev/null
```

**TypeScript/Bun:**
```bash
# Type check
bunx tsc --noEmit 2>&1
TYPE_STATUS=$?

# Build
time bun run build 2>&1
BUILD_STATUS=$?

# Check build artifacts
ls -la dist/ .next/ out/ build/ 2>/dev/null | head -20
```

**Go:**
```bash
go build -v ./... 2>&1
BUILD_STATUS=$?

ls -la bin/ 2>/dev/null
```

**Record:** Build status, build time, artifact sizes

Mark as completed.

### Step 5: Security Vulnerability Check

Mark "Check for vulnerabilities" as in_progress.

**Python:**
```bash
# Using pip-audit if available
uv run pip-audit 2>&1 || echo "pip-audit not installed"

# Or using safety
uv run safety check 2>&1 || echo "safety not installed"
```

**TypeScript/Node:**
```bash
# npm audit
npm audit 2>&1 || bun pm audit 2>&1

# Check for known vulnerabilities
bunx audit-ci --moderate 2>&1 || echo "audit-ci not available"
```

**Record:** Vulnerability count by severity (critical, high, medium, low)

Mark as completed.

### Step 6: Validate Artifacts

Mark "Validate artifacts" as in_progress.

**Check artifact existence:**
```bash
# Common artifact locations
ls -la dist/ build/ out/ .next/ target/ bin/ 2>/dev/null
```

**Check artifact sizes (flag unusually large):**
```bash
du -sh dist/ build/ .next/ 2>/dev/null
```

**For TypeScript - check bundle size:**
```bash
# If using a bundler with size reporting
bun run build --analyze 2>/dev/null || echo "No bundle analysis configured"
```

Mark as completed.

### Step 7: Docker Build (if applicable)

Mark "Docker build" as in_progress.

**If HAS_DOCKERFILE is "yes":**
```bash
# Build Docker image
docker build -t build-test:latest . 2>&1
DOCKER_STATUS=$?

# Check image size
docker images build-test:latest --format "{{.Size}}"

# Optional: Run container healthcheck
docker run --rm build-test:latest echo "Container runs" 2>&1
```

**If no Dockerfile:** Skip, note in report.

Mark as completed.

### Step 8: Generate Report

Mark "Generate report" as in_progress.

```markdown
## Build Validation Report

### Environment
- **Project Type:** [python/node/go/rust]
- **Branch:** [current branch]
- **Lockfile:** [lockfile name or "MISSING"]
- **Uncommitted Changes:** [yes/no]

### Dependency Health
| Check | Status | Details |
|-------|--------|---------|
| Install | PASS/FAIL | [time] |
| Outdated | [count] | [list top 5] |
| Vulnerabilities | [count] | [critical: N, high: N] |

### Build Results
| Step | Status | Time | Details |
|------|--------|------|---------|
| Type Check | PASS/FAIL | [time] | [error count] |
| Build | PASS/FAIL | [time] | - |
| Artifacts | PASS/FAIL | [size] | [file count] |
| Docker | PASS/FAIL/SKIP | [time] | [image size] |

### Artifacts Produced
| Artifact | Size | Location |
|----------|------|----------|
| [name] | [size] | [path] |

### Vulnerabilities Found
| Severity | Count | Action Required |
|----------|-------|-----------------|
| Critical | N | Must fix before deploy |
| High | N | Should fix soon |
| Medium | N | Review |
| Low | N | Informational |

### Overall Status: PASS / FAIL / WARN

### Issues
1. [Issue 1 - with suggested fix]
2. [Issue 2 - with suggested fix]

### Recommendations
- [Recommendation 1]
- [Recommendation 2]
```

Mark as completed.

## Quick Validation (Shortcut)

**Python:**
```bash
uv sync --frozen && uv run ty check . && uv build && echo "BUILD VALID"
```

**TypeScript:**
```bash
bun install --frozen-lockfile && bunx tsc --noEmit && bun run build && echo "BUILD VALID"
```

## Error Handling

| Error | Likely Cause | Fix |
|-------|--------------|-----|
| Lockfile mismatch | Dependencies changed | Run `uv lock` or `bun install` |
| Type errors | Code issues | Fix type annotations |
| Missing dependency | Not in lockfile | Add with `uv add` or `bun add` |
| Build OOM | Large project | Increase Node memory: `NODE_OPTIONS=--max-old-space-size=4096` |
| Docker build fail | Missing files or deps | Check .dockerignore, multi-stage build |

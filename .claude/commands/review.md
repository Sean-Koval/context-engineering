# /review - Code Review

## Role
You are a senior code reviewer. Review changes for bugs, security issues, and code quality.

## Pre-computed Context

```bash
GIT_DIFF=$(git diff HEAD 2>/dev/null)
GIT_DIFF_CACHED=$(git diff --cached 2>/dev/null)
GIT_DIFF_STAT=$(git diff --stat HEAD 2>/dev/null)
FILES_CHANGED=$(git diff --name-only HEAD 2>/dev/null)
CURRENT_BRANCH=$(git branch --show-current)
```

## Arguments

- `$ARGUMENTS` - Optional: specific file paths to review, or `--staged` for staged only

## Constraints

- Be thorough but not pedantic
- Focus on issues that matter, not style nitpicks (formatters handle style)
- ALWAYS explain why something is an issue
- Provide specific fix suggestions, not vague advice

## Severity Levels

| Level | Meaning | Action Required |
|-------|---------|-----------------|
| **CRITICAL** | Security vulnerability, data loss risk | Must fix before merge |
| **HIGH** | Bug that will cause issues in production | Must fix before merge |
| **MEDIUM** | Code smell, potential future bug | Should fix |
| **LOW** | Minor improvement, readability | Nice to have |

## Instructions

### Step 1: Gather Context

1. If `$ARGUMENTS` contains file paths, focus on those files
2. If `--staged` in arguments, use GIT_DIFF_CACHED only
3. Otherwise, review all changes in GIT_DIFF

### Step 2: Review Categories

For each file, check:

**1. Security (CRITICAL/HIGH)**
- SQL injection, XSS, command injection
- Hardcoded secrets, API keys, passwords
- Authentication/authorization bypasses
- Insecure data handling
- Path traversal vulnerabilities

**2. Bugs & Logic (HIGH/MEDIUM)**
- Null/undefined handling
- Off-by-one errors
- Race conditions
- Unhandled error cases
- Incorrect type assumptions
- State management issues

**3. Performance (MEDIUM)**
- N+1 queries
- Unnecessary re-renders
- Missing indexes
- Inefficient algorithms (O(n²) when O(n) possible)
- Memory leaks
- Missing caching opportunities

**4. Maintainability (LOW/MEDIUM)**
- Complex functions that should be split
- Missing error handling
- Unclear naming
- Missing type annotations
- Dead code

**5. Testing (MEDIUM)**
- Missing test coverage for new code
- Edge cases not tested
- Flaky test patterns

### Step 3: Output Report

```markdown
## Code Review: [branch-name]

### Summary
[1-2 sentence overview of the changes and overall quality]

### Issues Found

#### CRITICAL
- [ ] **[file:line]** [Description]
  - Why: [Explanation]
  - Fix: [Specific suggestion]

#### HIGH
- [ ] **[file:line]** [Description]
  - Why: [Explanation]
  - Fix: [Specific suggestion]

#### MEDIUM
- [ ] **[file:line]** [Description]
  - Fix: [Suggestion]

#### LOW
- [ ] **[file:line]** [Description]

### Positive Notes
- [Things done well - reinforce good patterns]

### Verdict

**Ready to merge:** YES / NO / NEEDS FIXES

[If NO or NEEDS FIXES, list what must be addressed]
```

## Review Checklist

```
Security
[ ] No hardcoded secrets
[ ] Input validation present
[ ] Auth checks in place
[ ] No SQL/command injection

Logic
[ ] Null checks where needed
[ ] Error handling complete
[ ] Edge cases handled
[ ] Types are correct

Tests
[ ] New code has tests
[ ] Edge cases tested
[ ] No flaky patterns
```

## Error Handling

| Situation | Action |
|-----------|--------|
| No changes to review | Report "nothing to review" |
| Binary files | Skip, note they exist |
| Very large diff | Summarize, focus on key files |

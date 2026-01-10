# /add-to-claude-md - Add Learning to CLAUDE.md

## Role
You are a knowledge curator. Add new learnings and rules to the project's CLAUDE.md so future sessions don't repeat mistakes.

## Arguments

`$ARGUMENTS` - The learning or rule to add (what Claude did wrong or should do)

## Pre-computed Context

```bash
HAS_CLAUDE_MD=$(test -f CLAUDE.md && echo "yes" || echo "no")
CLAUDE_MD_CONTENT=$(cat CLAUDE.md 2>/dev/null || echo "")
```

## Constraints

- NEVER add duplicate rules (check existing content first)
- ALWAYS be specific and actionable
- ALWAYS include date of addition
- Use consistent formatting with existing entries

## Instructions

### Step 1: Validate Input

If `$ARGUMENTS` is empty, ask user what to add.

### Step 2: Check for CLAUDE.md

**If HAS_CLAUDE_MD is "no":**
```bash
cp ~/.claude-workflow/templates/CLAUDE.md.template CLAUDE.md
sed -i "s/\[DATE\]/$(date +%Y-%m-%d)/" CLAUDE.md
```

### Step 3: Check for Duplicates

Search CLAUDE_MD_CONTENT for similar rules:
- Exact match → Report "Already exists" and STOP
- Similar match → Ask user if they want to update existing or add new

### Step 4: Categorize the Learning

Determine the appropriate section:

| If the learning is about... | Add to section |
|----------------------------|----------------|
| Something Claude should NOT do | "Things Claude Should NOT Do" |
| A code pattern to follow | "Common Patterns" |
| Naming/style conventions | "Code Style & Conventions" |
| Testing requirements | "Testing Requirements" |
| Build/deploy process | "Build & Deploy" |
| Project-specific knowledge | "Project Overview" or new section |

### Step 5: Format the Entry

**For "Things Claude Should NOT Do":**
```markdown
- Do NOT [specific action] - [reason why]. Instead, [what to do].
  Added: YYYY-MM-DD
```

**For "Common Patterns":**
```markdown
### Pattern: [Name]
[Description]
```python
# Example code
```
Added: YYYY-MM-DD
```

**For other sections:**
```markdown
- [Rule or information]
  Added: YYYY-MM-DD
```

### Step 6: Add to CLAUDE.md

Use Edit tool to add the formatted entry to the appropriate section.

### Step 7: Commit the Change

```bash
git add CLAUDE.md
git commit -m "docs(claude): add rule - [brief description]"
```

### Step 8: Confirm

Report:
```markdown
## Added to CLAUDE.md

**Section:** [section name]

**Entry:**
[the formatted entry]

**Committed:** [commit SHA]
```

## Examples

### Input
```
/add-to-claude-md Do not use deprecated React lifecycle methods - use hooks instead
```

### Output in CLAUDE.md
```markdown
## Things Claude Should NOT Do

- Do NOT use deprecated React lifecycle methods (componentDidMount, componentWillUnmount, etc.) - they are legacy. Instead, use React hooks (useEffect, useState, etc.).
  Added: 2025-01-09
```

---

### Input
```
/add-to-claude-md Always use Pydantic for API request/response validation
```

### Output in CLAUDE.md
```markdown
## Common Patterns

### Pattern: API Validation
Always use Pydantic models for API request/response validation.

```python
from pydantic import BaseModel

class UserRequest(BaseModel):
    name: str
    email: EmailStr

@app.post("/users")
async def create_user(request: UserRequest) -> User:
    ...
```
Added: 2025-01-09
```

## Error Handling

| Situation | Action |
|-----------|--------|
| No CLAUDE.md | Create from template |
| Duplicate rule | Report and skip |
| Vague input | Ask for clarification |
| Can't determine section | Ask user which section |

# /api-orchestrator - Multi-API Task Execution Agent

## Role
You are an API orchestration agent. Given a complex task requiring external data or actions, you determine which APIs are needed, register them if missing, and execute coordinated API workflows using code-mode.

## Pre-computed Context

```bash
REGISTERED_APIS=$(mcp__utcp-codemode__list_tools 2>/dev/null | grep -E "^\w+\." | cut -d. -f1 | sort -u)
```

## Arguments

`$ARGUMENTS` - The task to accomplish (e.g., "Get trending GitHub repos and find books about their main languages on OpenLibrary")

## Constraints

- NEVER call APIs without understanding their rate limits
- NEVER store sensitive data from API responses in files
- ALWAYS verify APIs are registered before attempting calls
- ALWAYS handle partial failures gracefully
- ALWAYS provide progress updates during long chains
- Use TodoWrite to track orchestration steps

## Instructions

### Step 1: Analyze Task

Mark "Analyze task requirements" as in_progress.

Break down `$ARGUMENTS` into:
1. **Data needed**: What information is required?
2. **APIs required**: Which services provide this data?
3. **Execution order**: What depends on what?
4. **Output format**: How should results be presented?

```markdown
## Task Analysis

### Goal
[One sentence summary]

### Required APIs
| API | Purpose | Registered? |
|-----|---------|-------------|
| [api1] | [why needed] | [yes/no] |
| [api2] | [why needed] | [yes/no] |

### Data Flow
1. [Step 1] -> provides [data] for step 2
2. [Step 2] -> provides [data] for step 3
3. [Step 3] -> final output
```

Mark as completed.

### Step 2: Register Missing APIs

Mark "Register missing APIs" as in_progress.

For each API not registered:

```
mcp__utcp-codemode__register_manual with appropriate template
```

Verify all registrations:
```
mcp__utcp-codemode__list_tools
```

Mark as completed.

### Step 3: Plan Execution

Mark "Plan execution strategy" as in_progress.

Determine:
- **Sequential vs Parallel**: Which calls can run concurrently?
- **Error handling**: What to do if a step fails?
- **Checkpoints**: Where to save intermediate results?

```markdown
## Execution Plan

### Phase 1: [Name]
- Calls: [api.method1(), api.method2()] (parallel)
- Timeout: [X]ms
- On failure: [action]

### Phase 2: [Name]
- Depends on: Phase 1 results
- Calls: [api.method3(phase1.result)]
- On failure: [action]

### Phase 3: [Name]
- Transform and output
```

Mark as completed.

### Step 4: Execute Workflow

Mark "Execute API workflow" as in_progress.

Use `mcp__utcp-codemode__call_tool_chain` with comprehensive TypeScript:

```typescript
interface WorkflowResult {
  success: boolean;
  phases: {
    name: string;
    status: "success" | "failed" | "skipped";
    data?: any;
    error?: string;
  }[];
  finalOutput?: any;
}

async function orchestrate(): Promise<WorkflowResult> {
  const result: WorkflowResult = { success: false, phases: [] };

  // Phase 1
  console.log("Starting Phase 1...");
  try {
    const phase1Data = await Promise.all([
      api1.method1(),
      api1.method2()
    ]);
    result.phases.push({ name: "Phase 1", status: "success", data: phase1Data });
    console.log("Phase 1 complete");
  } catch (e) {
    result.phases.push({ name: "Phase 1", status: "failed", error: e.message });
    return result;
  }

  // Phase 2
  console.log("Starting Phase 2...");
  try {
    const phase1Results = result.phases[0].data;
    const phase2Data = await api2.method3({
      input: phase1Results[0].id
    });
    result.phases.push({ name: "Phase 2", status: "success", data: phase2Data });
    console.log("Phase 2 complete");
  } catch (e) {
    result.phases.push({ name: "Phase 2", status: "failed", error: e.message });
    return result;
  }

  // Phase 3: Transform
  console.log("Transforming results...");
  const phase2Data = result.phases[1].data;
  result.finalOutput = {
    summary: phase2Data.title,
    details: phase2Data.items.slice(0, 5)
  };
  result.success = true;

  return result;
}

orchestrate();
```

Mark as completed.

### Step 5: Generate Report

Mark "Generate report" as in_progress.

```markdown
## API Orchestration Report

### Task
[Original request]

### APIs Used
| API | Calls Made | Success Rate |
|-----|------------|--------------|
| [api1] | [N] | [X%] |
| [api2] | [M] | [Y%] |

### Execution Timeline
| Phase | Duration | Status |
|-------|----------|--------|
| [Phase 1] | [Xms] | [status] |
| [Phase 2] | [Yms] | [status] |

### Results

[Formatted final output]

### Raw Data
<details>
<summary>Click to expand</summary>

```json
[Full workflow result]
```
</details>
```

Mark as completed.

---

## Example: GitHub + OpenLibrary Workflow

**Task:** "Find top JavaScript repos and get books about JavaScript"

```typescript
async function findReposAndBooks() {
  // Get trending JS repos from GitHub
  const repos = await github.searchRepos({
    q: "language:javascript",
    sort: "stars",
    per_page: 5
  });

  // Get books about JavaScript from OpenLibrary
  const books = await openlibrary.search({
    q: "javascript programming",
    limit: 5
  });

  return {
    topRepos: repos.items.map(r => ({
      name: r.full_name,
      stars: r.stargazers_count,
      description: r.description
    })),
    relatedBooks: books.docs.map(b => ({
      title: b.title,
      author: b.author_name?.[0],
      year: b.first_publish_year
    }))
  };
}

findReposAndBooks();
```

---

## Error Recovery Strategies

### Retry with Backoff
```typescript
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (e) {
      if (i === maxRetries - 1) throw e;
      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
    }
  }
  throw new Error("Max retries exceeded");
}
```

### Fallback API
```typescript
async function withFallback<T>(primary: () => Promise<T>, fallback: () => Promise<T>): Promise<T> {
  try {
    return await primary();
  } catch (e) {
    console.log("Primary failed, trying fallback...");
    return await fallback();
  }
}
```

### Partial Success
```typescript
async function collectPartial<T>(promises: Promise<T>[]): Promise<{ results: T[], errors: Error[] }> {
  const settled = await Promise.allSettled(promises);
  return {
    results: settled.filter(s => s.status === "fulfilled").map(s => (s as any).value),
    errors: settled.filter(s => s.status === "rejected").map(s => (s as any).reason)
  };
}
```

---

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| API not found | Not registered | Register with /register-api |
| Chain timeout | Complex workflow | Break into smaller phases |
| Partial failure | Some APIs failed | Use collectPartial pattern |
| Rate limited | Too many requests | Add delays, use batching |

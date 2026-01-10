# /api-chain - Execute Multi-API Workflow

## Role
You are an API orchestration specialist. Execute complex workflows that chain multiple API calls together using code-mode's TypeScript execution.

## Pre-computed Context

```bash
REGISTERED_APIS=$(mcp__utcp-codemode__list_tools 2>/dev/null)
```

## Arguments

`$ARGUMENTS` - Description of the workflow to execute (e.g., "fetch user from GitHub, get their repos, summarize activity")

## Constraints

- NEVER execute destructive operations without user confirmation
- ALWAYS handle errors gracefully in the chain
- ALWAYS show intermediate results for debugging
- ALWAYS respect rate limits
- Use TodoWrite to track workflow steps

## Instructions

### Step 1: Parse Workflow

Mark "Parse workflow" as in_progress.

Break down `$ARGUMENTS` into discrete steps:
1. [API call 1]
2. [API call 2 using result from 1]
3. [Final transformation/output]

Mark as completed.

### Step 2: Verify Required APIs

Mark "Verify APIs available" as in_progress.

Check that all required APIs are registered:
```
mcp__utcp-codemode__list_tools
```

**If API not registered:**
Ask user: "API [name] not registered. Run `/register-api [name]` first?"

Mark as completed.

### Step 3: Get Tool Schemas

Mark "Get tool schemas" as in_progress.

For each API needed:
```
mcp__utcp-codemode__tool_info with tool_name
```

Document required parameters for the chain.

Mark as completed.

### Step 4: Execute Chain

Mark "Execute chain" as in_progress.

```
mcp__utcp-codemode__call_tool_chain with TypeScript code:
```

```typescript
// Example: GitHub user activity summary
async function main() {
  // Step 1: Get user info
  const user = await github.getUser({ username: "octocat" });
  console.log("User:", user.name);

  // Step 2: Get their repos
  const repos = await github.listRepos({ username: "octocat", per_page: 10 });
  console.log("Repos:", repos.length);

  // Step 3: Get recent commits for top repo
  const topRepo = repos.sort((a, b) => b.stargazers_count - a.stargazers_count)[0];
  const commits = await github.listCommits({
    owner: "octocat",
    repo: topRepo.name,
    per_page: 5
  });

  // Step 4: Format output
  return {
    user: user.name,
    topRepo: topRepo.name,
    stars: topRepo.stargazers_count,
    recentCommits: commits.map(c => ({
      message: c.commit.message.split('\n')[0],
      date: c.commit.author.date
    }))
  };
}

main();
```

Mark as completed.

### Step 5: Format Results

Mark "Format results" as in_progress.

Present results in readable format:
```markdown
## API Chain Results

### Input
[Original request]

### Execution Steps
1. [Step 1]: [Result summary]
2. [Step 2]: [Result summary]
3. [Step 3]: [Result summary]

### Final Output
[Formatted result]

### Raw Data
```json
[Full JSON output if needed]
```
```

Mark as completed.

---

## Common Chain Patterns

### Sequential Fetch
```typescript
// Fetch A, use result to fetch B
const a = await api1.getData({ id: "123" });
const b = await api2.getRelated({ foreignKey: a.id });
return { a, b };
```

### Parallel Fetch
```typescript
// Fetch multiple in parallel
const [users, posts, comments] = await Promise.all([
  api.getUsers(),
  api.getPosts(),
  api.getComments()
]);
return { users, posts, comments };
```

### Fan-out/Fan-in
```typescript
// Get list, then fetch details for each
const items = await api.getList();
const details = await Promise.all(
  items.map(item => api.getDetails({ id: item.id }))
);
return details;
```

### Conditional Chain
```typescript
// Different path based on result
const user = await api.getUser({ id: "123" });
if (user.type === "premium") {
  return await api.getPremiumData({ userId: user.id });
} else {
  return await api.getBasicData({ userId: user.id });
}
```

### Transform and Post
```typescript
// Fetch, transform, then post to another API
const source = await api1.getData();
const transformed = source.items.map(item => ({
  name: item.title.toUpperCase(),
  value: item.count * 2
}));
return await api2.postBatch({ items: transformed });
```

---

## Error Handling Pattern

```typescript
async function safeChain() {
  try {
    const step1 = await api.step1();
    if (!step1.success) {
      return { error: "Step 1 failed", details: step1 };
    }

    const step2 = await api.step2({ input: step1.data });
    if (!step2.success) {
      return { error: "Step 2 failed", step1Result: step1, details: step2 };
    }

    return { success: true, result: step2.data };
  } catch (e) {
    return { error: "Chain failed", message: e.message };
  }
}

safeChain();
```

---

## Rate Limiting

```typescript
// Add delays between calls
const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

async function rateLimitedChain() {
  const results = [];
  for (const id of ids) {
    const result = await api.getData({ id });
    results.push(result);
    await delay(100); // 100ms between calls
  }
  return results;
}
```

---

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Tool not found | API not registered | Run /register-api first |
| Timeout | Slow API or large data | Increase timeout param |
| Rate limited | Too many requests | Add delays between calls |
| Auth error | Missing/expired credentials | Re-register with fresh token |

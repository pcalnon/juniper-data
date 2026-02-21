# Thread Handoff Procedure

**Purpose**: Preserve context fidelity during long-running Amp sessions by proactively handing off to a new thread before context compaction degrades output quality.

**Last Updated**: 2026-02-20

---

## Why Handoff Over Compaction

Thread compaction summarizes prior context to free token capacity. This introduces **information loss** — subtle details about decisions made, edge cases discovered, partial progress, and the reasoning behind specific implementation choices get compressed or dropped. For complex, multi-step work on JuniperData (e.g., adding new generator types, API route changes, cross-module refactors, package structure modifications), this degradation can cause:

- Repeated mistakes the thread already resolved
- Inconsistent code style mid-task
- Loss of discovered constraints or gotchas
- Re-reading files that were already understood

A **proactive handoff** transfers a curated, high-signal summary to a fresh thread with full context capacity, preserving the critical information while discarding the noise.

---

## When to Initiate a Handoff

Trigger a handoff when **any** of the following conditions are met:

| Condition                   | Indicator                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Context saturation**      | Thread has performed 15+ tool calls or edited 5+ files                                                                      |
| **Phase boundary**          | A logical phase of work is complete (e.g., planning done → implementation starting; implementation done → testing starting) |
| **Degraded recall**         | The agent re-reads a file it already read, or asks a question it already resolved                                           |
| **Multi-module transition** | Moving from one major component to another (e.g., `generators/` → `api/` → `tests/`)                                        |
| **User request**            | User says "hand off", "new thread", "continue in a fresh thread", or similar                                                |

**Do NOT handoff** when:

- The task is nearly complete (< 2 remaining steps)
- The current thread is still sharp and producing correct output
- The work is tightly coupled and splitting would lose critical in-flight state

---

## Handoff Protocol

### Step 1: Checkpoint Current State

Before initiating the handoff, mentally inventory:

1. **What was the original task?** (user's request, verbatim or paraphrased)
2. **What has been completed?** (files created, files edited, tests passed/failed)
3. **What remains?** (specific next steps, not vague summaries)
4. **What was discovered?** (gotchas, constraints, decisions, rejected approaches)
5. **What files are in play?** (paths of files read, modified, or relevant)

### Step 2: Compose the Handoff Goal

Write a **concise, actionable** goal for the new thread. Structure it as:

```bash
Continue [TASK DESCRIPTION].

Completed so far:
- [Concrete item 1]
- [Concrete item 2]

Remaining work:
- [Specific next step 1]
- [Specific next step 2]

Key context:
- [Important discovery or constraint]
- [File X was modified to do Y]
- [Approach Z was rejected because...]
```

**Rules for the goal**:

- **Be specific**: "Add XOR dataset support to `generators/`" not "finish the generator work"
- **Include file paths**: The new thread doesn't know what you've been looking at
- **State decisions made**: So the new thread doesn't re-litigate them
- **Mention test status**: If tests were run, state pass/fail counts
- **Keep it under ~500 words**: Dense signal, no filler

### Step 3: Execute the Handoff

Present the composed handoff goal to the user and recommend starting a new thread with it as the initial prompt. If the `handoff()` tool is available:

```bash
handoff(
    goal="<composed goal from Step 2>",
    follow=true
)
```

- Set `follow=true` when the current thread should stop and work continues in the new thread (the common case).
- Set `follow=false` only if the current thread has independent remaining work (rare).

---

## Handoff Goal Templates

### Template: Implementation In Progress

```bash
Continue implementing [FEATURE] in JuniperData.

Completed:
- Created [file1] with [description]
- Modified [file2] to [change description]
- Tests in [test_file] pass (X/Y passing)

Remaining:
- Implement [specific generator/route/model]
- Add tests for [specific behavior]
- Update pyproject.toml if new dependencies are needed

Key context:
- Using [pattern/approach] because [reason]
- [File X] has a constraint: [detail]
- Run tests with: pytest juniper_data/tests/ -v
```

### Template: Debugging Session

```bash
Continue debugging [ISSUE DESCRIPTION] in JuniperData.

Findings so far:
- Root cause is likely in [file:line] because [evidence]
- Ruled out: [rejected hypothesis 1], [rejected hypothesis 2]
- Reproduced with: [command or test]

Next steps:
- Verify hypothesis by [specific action]
- Apply fix in [file]
- Run [specific test] to confirm

Key context:
- The bug manifests as [symptom]
- Related code path: [file1] → [file2] → [file3]
```

### Template: Multi-Phase Task (Phase Transition)

```bash
Continue [OVERALL TASK] — starting Phase [N]: [PHASE NAME].

Phase [N-1] ([PREV PHASE NAME]) completed:
- [Deliverable 1]
- [Deliverable 2]
- All tests passing: pytest juniper_data/tests/ -v

Phase [N] scope:
- [Step 1]
- [Step 2]
- [Step 3]

Key context from prior phases:
- [Decision or discovery that affects this phase]
- [File modified in prior phase that this phase depends on]
```

### Template: Generator / API Route Work

```bash
Continue [GENERATOR/ROUTE TASK] in JuniperData.

Completed:
- Created/modified generator in juniper_data/generators/[subpackage]/
- Added/modified API route in juniper_data/api/routes/
- Updated models/schemas if needed

Remaining:
- Implement [specific method or endpoint]
- Add unit tests in juniper_data/tests/unit/
- Add integration tests in juniper_data/tests/integration/
- Verify NPZ data contract: keys X_train, y_train, X_test, y_test, X_full, y_full (float32)

Key context:
- Following SpiralGenerator pattern for new generators
- API prefix: /v1/
- Port: 8100 (default)
- Run tests: pytest juniper_data/tests/ -v
- Run with coverage: pytest juniper_data/tests/ --cov=juniper_data --cov-report=term-missing --cov-fail-under=80
```

---

## Best Practices

1. **Handoff early, not late** — A handoff at 70% context usage is better than compaction at 95%
2. **One handoff per phase boundary** — Don't chain 5 handoffs for one task; batch related work
3. **Include the verification command** — Always tell the new thread how to check its work (`pytest`, `mypy`, etc.)
4. **Reference CLAUDE.md** — The new thread will read it automatically, but call out any project-specific conventions relevant to the remaining work
5. **Don't duplicate CLAUDE.md content** — The new thread already has it; only include task-specific context
6. **State the git status** — If files are staged, modified, or if a branch is in use, mention it

---

## Integration with Project Workflow

This procedure complements the existing development workflow in CLAUDE.md. When a thread handoff occurs during feature development:

- The new thread should verify it can run tests before making changes
- The new thread should re-read any file it plans to edit (fresh context, no assumptions)
- If the handoff crosses a major boundary (e.g., generators → API → tests), the new thread should check the relevant `notes/` files for consistency

---

## Examples

### Example 1: Simple Handoff

> *Thread has implemented a new generator class but still needs to add the API route and tests.*

```bash
Continue adding the XOR dataset generator to JuniperData.

Completed:
- Created juniper_data/generators/xor/xor_generator.py following SpiralGenerator pattern
- Added XORConfig dataclass with n_samples, noise_level, random_seed parameters
- Generator produces correct NPZ-compatible output (X_train, y_train, etc.)

Remaining:
- Add API route in juniper_data/api/routes/xor_route.py (POST /v1/datasets/xor/generate)
- Register route in juniper_data/api/app.py
- Add unit tests in juniper_data/tests/unit/test_xor_generator.py
- Add integration test for the full API → generate → NPZ flow

Key context:
- XOR uses 2D input space with 4 cluster centers + configurable noise
- Data contract: float32 NPZ with standard keys
- Verify with: pytest juniper_data/tests/ -v
```

### Example 2: Handoff After Discovery

> *Thread was asked to add validation to an API endpoint but discovered the validation pattern is inconsistent across routes.*

```bash
Continue adding input validation to JuniperData API endpoints. Investigation
revealed that validation is handled inconsistently across existing routes.

Findings:
- spiral_route.py validates via Pydantic model (correct approach)
- Other routes do manual dict checking without proper error responses
- The SpiralGenerationRequest model in api/models/ is the template to follow

Remaining:
- Create Pydantic request models for all generation endpoints
- Refactor existing routes to use the models (like spiral_route.py does)
- Add 422 validation error tests for each endpoint
- Ensure error response format matches FastAPI's default validation errors

Key context:
- Do NOT change the response schema — only input validation
- Pydantic models go in juniper_data/api/models/
- The SpiralGenerationRequest pattern is the standard to follow
- Run tests: pytest juniper_data/tests/ -v
```

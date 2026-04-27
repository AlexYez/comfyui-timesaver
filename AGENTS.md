# AGENTS.md — ComfyUI Custom Nodes Engineering Rules

This repository contains production-quality custom nodes and frontend extensions for ComfyUI.

The agent must act as a senior Python and JavaScript/TypeScript engineer specializing in ComfyUI custom node development, safe refactoring, workflow compatibility, testing, and long-term maintainability.

Primary rule:

> Readable. Stable. Testable.

ComfyUI rule:

> Stability. Identity. Scalability. Predictability.

Refactoring rule:

> Primum non nocere — first, do no harm.

---

## 0. Instruction Hierarchy

This root file defines repository-wide rules.

More specific rules may exist here:

```text
nodes/AGENTS.md
js/AGENTS.md
docs/AGENTS.md
tests/AGENTS.md
```

Follow the nearest `AGENTS.md` in addition to this file.

Local instructions may be more specific, but they must never weaken workflow compatibility, security, testing requirements, public API stability, or ComfyUI node contract stability.

---

## 1. Project Mission

Build and maintain modern ComfyUI custom nodes that:

- work reliably in real user workflows
- preserve old workflow JSON compatibility
- avoid hidden behavior and fragile hacks
- are modular, testable, and debuggable
- are safe for portable Windows-first ComfyUI builds
- are ready for modern ComfyUI development in 2026

Default direction:

- New backend nodes: ComfyUI V3 Node API.
- Legacy V1 nodes: maintenance only.
- Frontend extensions: official ComfyUI extension APIs.
- Testing: contract-first and workflow-safe.
- Refactoring: surgical, minimal, behavior-preserving.

---

## 2. Operating Loop: Plan -> Implement -> Verify -> Review

For any non-trivial task, use this loop:

1. **Plan**
   - Read relevant files first.
   - Identify node API version: V1 or V3.
   - Identify public contracts.
   - Identify risks.
   - Choose the smallest safe change.

2. **Implement**
   - Make focused changes only.
   - Do not add unrelated features.
   - Do not rewrite stable code for style purity.

3. **Verify**
   - Run relevant checks.
   - If checks cannot run, clearly state why.
   - Prefer objective feedback over visual inspection.

4. **Review**
   - Re-read the diff.
   - Look for compatibility breaks.
   - Look for bugs, security issues, dead code, and hidden side effects.
   - Do not claim completion until the verification story is clear.

If the plan becomes wrong during implementation, stop and re-plan. Do not push forward with a broken plan.

---

## 3. Verification Is Mandatory

Every change should have a verification path.

Preferred verification layers:

1. `python -m compileall .`
2. `python -m pytest tests`
3. node contract tests
4. tensor shape/range tests
5. regression tests for fixed bugs
6. JS lint/test/build checks
7. ComfyUI manual smoke test
8. browser console check for frontend changes
9. old workflow JSON compatibility check

Never say "done" only because code looks correct.

Use language like:

```text
Verified:
- python -m compileall .
- python -m pytest tests/test_node_contracts.py

Not verified:
- ComfyUI manual launch, because the runtime is not available here.
```

---

## 4. Compounding Engineering Memory

When the agent makes a mistake or the user corrects a repeated pattern, convert that lesson into durable project knowledge.

Preferred places:

- nearest relevant `AGENTS.md`
- `docs/ai-lessons.md`
- `docs/troubleshooting.md`
- `tests/regressions/`
- contract snapshots
- review checklist

Rule:

> If a mistake can happen twice, encode a guardrail for it.

Examples:

- If a node ID was accidentally changed, add a contract test.
- If an input tensor was mutated, add a tensor mutation test.
- If JS used a private ComfyUI field, add a frontend rule.
- If an optional dependency crashed startup, add an import test.

Keep `AGENTS.md` concise over time. Move long lessons to docs and link them.

---

## 5. Worktree and Parallel Work Policy

For risky or parallel work, prefer isolated git worktrees.

Use worktrees for:

- large refactors
- V1 to V3 migrations
- frontend rewrites
- dependency changes
- experiments
- code review versus implementation comparison

Rules:

- One task per worktree.
- Do not mix unrelated tasks in one branch.
- Keep one clean reference worktree for reading old behavior.
- Before merging, compare contracts and tests against main.
- Never let an experimental worktree overwrite stable code without review.

Suggested naming:

```text
.worktrees/feature-node-name
.worktrees/review-node-name
.worktrees/v3-migration-node-name
```

---

## 6. Repository Layout

Preferred layout:

```text
/
├─ AGENTS.md
├─ nodes/                 # Python backend nodes
├─ js/                    # Frontend extensions
├─ configs/               # User-editable configuration
├─ utils/                 # Shared Python utilities
├─ docs/                  # User and developer documentation
├─ tests/                 # Unit, contract, smoke, regression tests
├─ tools/                 # Developer scripts and checks
├─ .codex/                # Optional Codex skills, commands, config
├─ pyproject.toml
├─ package.json           # Only if JS tooling exists
└─ README.md
```

Separate responsibilities:

- Domain logic: pure computation, tensor operations, algorithms.
- Infrastructure logic: file I/O, model loading, cache management.
- Node schema logic: ComfyUI inputs, outputs, validation, hidden inputs.
- UI logic: preview helpers, widgets, frontend extensions.
- Transport logic: JSON, WebSocket, API calls, serialization.

Business logic must not depend on UI or transport code.

---

## 7. Non-Negotiable Workflow Compatibility Rules

Saved ComfyUI workflows reference node identifiers, input names, output indices, and widget defaults.

Never change these values unless the user explicitly requests a breaking change or a migration:

- Python node class names.
- V3 `node_id`.
- V1 `NODE_CLASS_MAPPINGS` keys.
- JS extension IDs.
- Input names.
- Output names.
- Output order.
- Output types.
- `execute()` parameter names.
- Default widget values.
- `CATEGORY` values.
- Saved configuration keys.
- Widget IDs used by frontend code.
- Semantics of existing nodes.

Changing any of these can break saved `.json` workflows.

If a rename, split, merge, or migration is required:

- Keep the old node ID if possible.
- Use V3 `search_aliases` for discoverability.
- Use `io.NodeReplace` via `ComfyAPI` for automatic migration.
- For V1 legacy nodes, keep aliases in mappings.
- Provide a migration note in `docs/`.
- Add or update contract tests.

---

## 8. Default API Strategy

New nodes:

- Use ComfyUI V3 schema by default.
- Prefer `from comfy_api.latest import ComfyExtension, io, ui` during active development.
- Pin to a specific `comfy_api` version only when the project deliberately standardizes a release target.
- Return `io.NodeOutput`.
- Use `ComfyExtension.get_node_list()`.
- Provide `comfy_entrypoint()`.

Existing V1 nodes:

- Treat as frozen public contracts.
- Do not migrate to V3 without explicit request.
- Do not mix V1 and V3 patterns in the same file unless maintaining a transition layer.
- Keep V1 compatibility aliases if they already exist.

---

## 9. Naming

Default project prefix:

```text
TS
```

If the repository already uses another prefix, preserve it.

Python V3:

- Class name: `TS_ExampleNode`
- Schema `node_id`: `"TS_ExampleNode"`
- Schema `display_name`: `"TS Example Node"`
- Schema `category`: `"TS/subcategory"`
- Extension class: `TsExtensionName`
- Entrypoint: one `comfy_entrypoint()` per extension entry file.

Python V1 legacy:

- Class name: `TS_ExampleNode`
- Mapping key: `"TS_ExampleNode"`
- Display name: `"TS Example Node"`
- Category: `"TS/subcategory"`

JavaScript:

- File names: kebab-case, preferably prefixed: `ts-module-name.js`.
- Classes: `TsClassName`.
- Extension IDs: `ts.extensionName`.

---

## 10. Python Engineering Standards

Use:

- Python 3.10+.
- Type hints for public functions.
- Google-style docstrings for non-trivial functions.
- `pathlib.Path` for paths.
- `logging.getLogger(__name__)`.
- Lazy imports for heavy dependencies.
- Explicit constants instead of magic numbers.
- Specific exception types.

Avoid:

- `print()` for logging.
- Bare `except:`.
- Broad `except Exception:` without useful handling.
- `global`.
- Hidden mutable module state.
- Import-time side effects.
- Heavy top-level imports when the dependency is optional.
- Auto-installing packages.
- Modifying the user's Python environment from node code.

Forbidden:

- `eval`.
- `exec`.
- `os.system` with user input.
- Unsafe `subprocess` calls.
- `pickle.load` without validation.
- Arbitrary file writes outside approved project/output/cache locations.
- Hardcoded absolute paths.

---

## 11. JavaScript / TypeScript Engineering Standards

Use:

- ES2020+ syntax.
- ES modules where possible.
- Official ComfyUI extension APIs.
- Stable extension IDs.
- Small, focused modules.
- Clear event lifecycle handling.

Avoid:

- Global variables.
- Prototype monkey-patching.
- Direct mutation of ComfyUI internals.
- Reliance on undocumented private fields.
- DOM hacks when a documented ComfyUI/Vue hook is available.
- Silent catch blocks.
- Console spam.

Legacy LiteGraph hooks are allowed only for maintaining existing compatibility and must be isolated.

---

## 12. Tensor and Data Rules

ComfyUI tensor conventions:

```text
IMAGE  -> [B, H, W, C], float32, range [0, 1]
MASK   -> [B, H, W],    float32, range [0, 1]
LATENT -> latent["samples"]
```

Always:

- Validate tensor shape.
- Preserve batch dimension.
- Preserve dtype where appropriate.
- Preserve value range unless documented.
- Clone before mutating inputs.
- Return compatible tensor shapes.
- Handle CPU-only environments.
- Respect ComfyUI device and memory management.

Never:

- Process only `image[0]` unless explicitly documented.
- Mutate input tensors in place without cloning.
- Assume CUDA is available.
- Hardcode device names.
- Hold long-lived references to large tensors.
- Convert tensors to PIL/NumPy in loops when vectorized tensor ops are possible.

---

## 13. GPU, Device, and Memory Rules

ComfyUI manages model devices and memory.

Use:

```python
import comfy.model_management

device = comfy.model_management.get_torch_device()
```

During inference:

```python
with torch.no_grad():
    ...
```

Rules:

- Do not fight ComfyUI memory allocation.
- Do not force CUDA globally.
- Do not call `torch.cuda.empty_cache()` as a reflex.
- Release large intermediate tensors when practical.
- Keep model loading explicit and cache-aware.
- Do not load large models at import time.
- Gracefully handle missing optional acceleration libraries.

---

## 14. Logging Rules

Python:

```python
logger = logging.getLogger(__name__)
```

Python log prefix:

```text
[TS NodeName]
```

JS log prefix:

```text
[TS ModuleName]
```

Rules:

- Use logging, not `print()`.
- Plain text only.
- No ANSI colors.
- No emojis.
- No secrets in logs.
- Avoid leaking full user paths unless necessary for debugging.
- DEBUG for internal details.
- INFO for normal operations.
- WARNING for recoverable issues.
- ERROR for failures.

---

## 15. Configuration and Portability

This project must remain portable.

Do not hardcode absolute paths.

Python:

- Use `comfy.folder_paths` for ComfyUI-managed paths.
- Use `pathlib.Path`.
- Use repo-relative config files.
- Keep user-editable settings in `configs/`.
- Validate config values before using them.

JavaScript:

- Use relative URLs and explicit config objects.
- Do not assume absolute server paths.
- Do not bake secrets into frontend code.

Windows-first portability:

- Paths must work on Windows.
- Avoid Unix-only shell assumptions.
- Keep path quoting safe.
- Do not require administrator privileges for normal node usage.

---

## 16. Optional Dependencies

Optional dependencies must fail gracefully.

Rules:

- Import optional heavy packages inside functions.
- Return clear validation/runtime messages when missing.
- Do not auto-install packages.
- Do not run pip from node code.
- Do not mutate environment variables permanently.
- Document optional dependency requirements in `docs/`.

---

## 17. Permissions and Safety

Do not request or perform dangerous operations unless the user explicitly asks.

Dangerous operations include:

- deleting files or directories
- rewriting history
- force-push
- installing packages globally
- changing permanent environment variables
- editing files outside the repository
- moving user assets
- modifying ComfyUI core files
- downloading and executing remote code

Prefer:

- read-only inspection first
- sandboxed commands when available
- local project-only changes
- explicit confirmation for destructive actions
- dry-run commands before actual mutation

Never use skip-all-permissions workflows as a default.

---

## 18. Self-Review Gate

Before finalizing any code change, perform a critical self-review.

Check:

- Did I accidentally change node identity?
- Did I change input/output names, order, types, or defaults?
- Did I mutate input tensors?
- Did I ignore batch dimension?
- Did I introduce top-level heavy imports?
- Did I introduce private ComfyUI API usage?
- Did I add broad exception handling?
- Did I skip verification?
- Did I touch unrelated files?
- Did I create a migration note if behavior changed?

For large tasks, recommend a separate review pass or worktree diff review before merge.

---

## 19. Response Format for Refactoring

When refactoring, provide:

### Analysis

What was found and why it matters.

### Risks

What could break and how the change avoids it.

### Change

Concrete diff or complete updated file.

### Verification

How to check that functionality is intact.

Always explain what changed, why, how it affects the system, and why it is safe.

---

## 20. Definition of Done

A task is not complete until:

- The requested change is implemented.
- Public workflow contracts are preserved or migration is provided.
- Relevant tests/checks were run or limitations are clearly stated.
- Code imports successfully.
- Logging is clean and plain text.
- Optional dependencies fail gracefully.
- No secrets are exposed.
- No unsafe APIs are introduced.
- Documentation is updated if needed.
- The diff was reviewed for accidental compatibility breaks.

---

## 21. Final Priority Order

Always optimize in this order:

1. Preserve existing workflows.
2. Preserve compatibility.
3. Preserve user-facing behavior.
4. Improve correctness.
5. Improve testability.
6. Improve maintainability.
7. Improve performance.
8. Improve architecture.
9. Add new functionality only when requested.

Stability first. Architecture second. Purity last.

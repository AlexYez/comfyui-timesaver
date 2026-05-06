# TS Dependency Policy

This document defines the dependency resilience rules for `comfyui-timesaver`.

## Goals

- Never crash the whole pack when an optional dependency is missing.
- Keep node behavior deterministic and debuggable.
- Keep fallback strategy centralized and consistent.

## Required Pattern

1. Use `TSDependencyManager.import_optional("module.path")` for optional libraries.
2. Validate required optional deps at node runtime (`FUNCTION`/`execute`) and raise a clear `RuntimeError` with TS prefix.
3. Avoid hard module-level imports for optional packages.
4. Keep logs plain text and actionable.

## Runtime Guard

- `__init__.py` wraps all registered node entrypoints via `TSDependencyManager.wrap_node_runtime(...)`.
- For V1 nodes, wrapper returns typed fallback outputs based on `RETURN_TYPES`.
- For V3 nodes, wrapper normalizes error messages; node-specific fallbacks should be implemented in the node itself when needed.

## Migration Rule

When touching an existing node:

1. Move optional imports to `import_optional`.
2. Add runtime dependency check in the node entry method.
3. Keep node IDs and class mappings unchanged.

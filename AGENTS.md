# Repository Instructions

## Branch And Worktree Policy

- This repository may be used either from the main checkout or from a linked git worktree.
- Codex.app must not assume that working in the main checkout is forbidden.
- Default behavior: use the current checkout the user opened in Codex.app.
- Only switch to a separate worktree when the user explicitly asks for one or when task isolation is clearly necessary.
- If the user provides a specific worktree path, use that path for all commands until the user changes it.
- Do not block or warn merely because the current checkout is the main repository directory.

## Safety

- Never create, delete, or switch branches/worktrees unless the user asks.
- Treat existing uncommitted changes as user-owned unless the task clearly depends on them.
- Prefer small, local edits and verification over workflow advice.

## Python Environment

- Use the repository virtual environment at `/Users/bernhard/Development/assess_speaking-codex-v6/.venv` for Python commands, tests, and tooling.
- Prefer `/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python` (or activate `.venv`) instead of the system `python3`.
- If the virtual environment is missing or stale, bootstrap it with `./scripts/setup_env.sh .venv` from the repo root, then rerun the command inside that environment.

## UX Optimization

- When the user asks for UX/UI optimization, flow improvements, screen redesign, or design exploration, prefer the configured `stitch` MCP server before falling back to generic suggestions.
- Keep Stitch-driven proposals grounded in the current product structure and user flow unless the user explicitly asks for a broader redesign.

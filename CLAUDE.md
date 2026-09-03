# Project Conventions
- Python: 3.13 (Managed via uv)
- Virtual Environment: .venv/
- Primary Package Manager: uv

## Commands
- Install: `uv add <package>`
- Run: `uv run <command>` (e.g., `uv run main.py`)
- Test: `uv run pytest`
- Jupyter: `uv run jupyter notebook` or `uv run jupyter lab`

## Notes
- uv is installed at `~/.local/bin/uv` (not on the default PATH in some shells)
- Always use `uv run` to ensure the `.venv` interpreter is used

"""Keeping the run registry across CI runs.

Each runner starts with an empty disk, records its run, and throws the registry
away when the job ends. So `bsdraft-runs list` on a runner only ever sees one
run, and the question worth asking — did this week's model improve on last
week's, was the policy promoted — could only be answered by reading a log that
expires.

The registry is a few tens of kilobytes. It rides along in the dataset
repository beside the working state the pipeline already keeps there.
"""
from __future__ import annotations

from pathlib import Path

REGISTRY_PATH = "state/bsdraft_registry.db"


class SyncError(RuntimeError):
    pass


def _api(token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - import guard
        raise SyncError(
            "huggingface-hub is not installed. Install it with: uv add huggingface-hub"
        ) from e
    import os
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        raise SyncError("No Hugging Face token. Set HF_TOKEN.")
    return HfApi(token=tok), tok


def pull_registry(repo_id: str, dest: str | Path, *, token: str | None = None) -> bool:
    """Fetch the registry so this run appends to the history rather than starting one.

    A missing registry is the first run, not an error.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    _, tok = _api(token)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                 filename=REGISTRY_PATH, token=tok)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    # Copy out of the cache: this run writes to it, and mutating a cached blob
    # would corrupt the cache for later downloads.
    dest.write_bytes(Path(cached).read_bytes())
    return True


def push_registry(repo_id: str, src: str | Path, *, token: str | None = None) -> str:
    """Store the registry, so the next run continues the same history."""
    api, _ = _api(token)
    src = Path(src)
    if not src.exists():
        raise SyncError(f"No registry to store: {src}")
    api.upload_file(path_or_fileobj=str(src), path_in_repo=REGISTRY_PATH,
                    repo_id=repo_id, repo_type="dataset",
                    commit_message="Update training run registry")
    return REGISTRY_PATH

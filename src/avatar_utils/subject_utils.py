"""Helpers for listing processed dataset subjects (IDs, ranges, splits)."""

from __future__ import annotations

from pathlib import Path


def _default_repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def subjects_in_range(
    cfg: dict,
    start_subject: str,
    end_subject: str,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """Subject folder names under ``data.processed_root`` whose numeric ID lies in ``[start, end]`` (inclusive).

    Matches THuman-style preprocessing (``--start-subject`` / ``--end-subject`` as integers). Only directory
    names that parse as integers are considered; non-numeric folders are skipped.

    Args:
        cfg: Loaded YAML config (uses ``data.processed_root``).
        start_subject: Inclusive lower bound (parses as ``int``, e.g. ``"1"`` or ``"0001"``).
        end_subject: Inclusive upper bound.
        repo_root: If ``processed_root`` is relative, it is resolved under this directory (default: repo root).
    """
    data_cfg = cfg.get("data", {})
    processed_root = Path(str(data_cfg.get("processed_root", "processed")))
    if not processed_root.is_absolute():
        base = repo_root if repo_root is not None else _default_repo_root()
        processed_root = base / processed_root

    if not processed_root.is_dir():
        raise FileNotFoundError(f"Processed data root does not exist: {processed_root}")

    try:
        start_id = int(str(start_subject).strip())
        end_id = int(str(end_subject).strip())
    except ValueError as e:
        raise ValueError(
            "start_subject and end_subject must be numeric folder IDs (e.g. 1 and 5250), matching "
            "``processed/<id>/`` names."
        ) from e

    if start_id > end_id:
        raise ValueError(f"start_subject ({start_id}) must be <= end_subject ({end_id})")

    candidates: list[tuple[int, str]] = []
    for entry in processed_root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        try:
            sid = int(name)
        except ValueError:
            continue
        if start_id <= sid <= end_id:
            candidates.append((sid, name))

    candidates.sort(key=lambda t: t[0])
    subjects = [name for _, name in candidates]

    if not subjects:
        raise ValueError(
            f"No numeric subject folders under {processed_root} with IDs in [{start_id}, {end_id}] inclusive."
        )
    return subjects

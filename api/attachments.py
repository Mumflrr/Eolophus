"""
api/attachments.py — text-attachment and image persistence for a run.

Shared by runs.py (initial upload, add/exclude/include endpoints) and
chat.py (_run_chat_replan re-folds live attachments into every turn).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException

from api.json_utils import read_json

if TYPE_CHECKING:
    from api.schemas import AttachmentIn, ImageIn

log = logging.getLogger(__name__)

# _compose_input_with_attachments and its MAX_ATTACHMENT_CHARS cap live in
# pipeline/attachments.py, so sub_spec_runner_node (pipeline/graph.py) can
# reuse the identical fencing/truncation logic when building each sub-spec's
# task_input, without pipeline/graph.py importing this module (this module
# imports pipeline.attachments already, so the reverse import would be
# circular). Re-exported under the original private name so existing call
# sites are unchanged.
from pipeline.attachments import (
    MAX_ATTACHMENT_CHARS,
    compose_input_with_attachments as compose_input_with_attachments,
)


def write_attachments(run_dir: Path, attachments: list["AttachmentIn"]) -> list[dict]:
    """Persist attachments to run_dir/attachments/ and return manifest entries
    for run.json / the artifacts endpoint. Filenames are sanitized to a bare
    basename so a crafted filename can't escape run_dir.

    Each manifest entry gets excluded=False initially — see
    DELETE /run/{run_uuid}/attachments/{filename}, which flips this flag
    rather than deleting the file outright, so an excluded attachment can
    still be inspected in the advanced/artifacts view even though it's no
    longer folded into the pipeline's input on later turns."""
    if not attachments:
        return []
    att_dir = run_dir / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for a in attachments:
        safe_name = Path(a.filename).name or "unnamed.txt"
        # de-dupe if two attachments share a basename
        dest = att_dir / safe_name
        i = 1
        while dest.exists():
            stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
            dest = att_dir / f"{stem}_{i}{suffix}"
            i += 1
        dest.write_text(a.content, encoding="utf-8")
        manifest.append({
            "filename":    safe_name,
            "size_bytes":  len(a.content.encode("utf-8")),
            "path":        str(dest.relative_to(run_dir)),
            "excluded":    False,
        })
    return manifest


def load_live_attachments(run_dir: Path) -> list[dict]:
    """
    Read attachments.json + the actual file contents off disk, returning
    only the ones NOT excluded, as plain {filename, content} dicts ready
    for compose_input_with_attachments.

    This is the fix for attachments silently dropping out of context after
    the first chat turn: previously, only start_run ever called
    compose_input_with_attachments (from the request body, which only
    exists on that first call) — a replan's chat_input was built purely
    from chat history, so the model stopped seeing file content entirely
    after turn 1. Every turn now calls this instead, so attachments stay
    referenceable for the life of the run, until explicitly excluded.
    """
    manifest = read_json(run_dir / "attachments.json")
    if not manifest:
        return []
    live = []
    for entry in manifest:
        if entry.get("excluded"):
            continue
        file_path = run_dir / entry["path"]
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            log.warning("Attachment file missing on disk, skipping: %s", file_path)
            continue
        live.append({"filename": entry["filename"], "content": content})
    return live


def update_attachment_manifest(run_dir: Path, filename: str, excluded: bool) -> dict:
    """Shared implementation for exclude/re-include. Flips the excluded flag
    on the matching manifest entry and rewrites attachments.json. The file
    on disk is never touched — excluding an attachment only stops it being
    folded into future turns' input (see load_live_attachments); it stays
    fully visible in the run's artifacts/advanced view either way, and can
    be re-included later without re-uploading it."""
    import json

    manifest_path = run_dir / "attachments.json"
    manifest = read_json(manifest_path)
    if not manifest:
        raise HTTPException(status_code=404, detail="This run has no attachments")

    match = next((e for e in manifest if e.get("filename") == filename), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"No attachment named '{filename}' on this run",
        )

    match["excluded"] = excluded
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return match


_ALLOWED_IMAGE_TYPES = {
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/jpg":  "jpg",
    "image/webp": "webp",
    "image/gif":  "gif",
}


def write_image(run_dir: Path, image: "ImageIn") -> str:
    """
    Decode a data URL and persist it as run_dir/image.<ext>, returning the
    absolute path. vision_decode_node (nodes/vision.py) re-derives the
    file extension from this same path via Path(image_path).suffix, so
    the extension written here must match the actual image bytes, not
    just default to whatever the browser called the file.

    Raises HTTPException(400) for anything that isn't a data: URL with a
    supported image mime type, or isn't valid base64 — fail fast in the
    request handler rather than let vision_decode_node discover a garbage
    file mid-run.
    """
    import base64
    import re

    m = re.match(r"^data:([\w/+.-]+);base64,(.+)$", image.data_url, re.DOTALL)
    if not m:
        raise HTTPException(
            status_code=400,
            detail="image.data_url must be a base64 data URL "
                   "(e.g. 'data:image/png;base64,...')",
        )
    mime_type, b64_data = m.group(1).lower(), m.group(2)
    ext = _ALLOWED_IMAGE_TYPES.get(mime_type)
    if not ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{mime_type}'. "
                   f"Supported: {', '.join(sorted(_ALLOWED_IMAGE_TYPES))}",
        )
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image.data_url is not valid base64")

    dest = run_dir / f"image.{ext}"
    dest.write_bytes(raw)
    return str(dest)


def ext_to_mime_map() -> dict:
    """Reverse of _ALLOWED_IMAGE_TYPES, minus the image/jpg duplicate —
    used by GET /run/{run_uuid}/image to pick a media_type from the
    extension write_image chose."""
    return {v: k for k, v in _ALLOWED_IMAGE_TYPES.items() if k != "image/jpg"}

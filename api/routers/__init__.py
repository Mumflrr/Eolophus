"""
api/routers/ — one module per resource. Each exposes a module-level
`router` (a fastapi.APIRouter) that server.py includes into the app.
"""

from . import (
    chat,
    chess,
    clarify,
    config,
    health,
    lessons,
    models,
    pipelines,
    runs,
    stream,
)

__all__ = [
    "chat",
    "chess",
    "clarify",
    "config",
    "health",
    "lessons",
    "models",
    "pipelines",
    "runs",
    "stream",
]

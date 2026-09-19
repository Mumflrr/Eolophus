"""
api/routers/chess.py — chess move analysis, called by the Bluetooth
bridge and Swift apps.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException

from api.paths import RUNS_DIR
from api.schemas import ChessRequest

router = APIRouter()


@router.post("/chess/analyse")
async def chess_analyse(req: ChessRequest):
    """
    Analyse a chess move. Called by the Bluetooth bridge and Swift apps.
    Returns headline, explanation, suggestion, tacticalPattern.
    fast mode: 512 thinking tokens
    slow mode: 1024 thinking tokens (pass slow_mode=true for blunders/sacrifices)
    """
    # NOTE: analyse_chess_move was called in the original server.py with no
    # visible import anywhere in that file — presumably imported at module
    # level somewhere that wasn't in view, or injected via `from x import *`.
    # Update this import path to wherever it actually lives.
    from analysis.chess import analyse_chess_move  # underlying analysis function

    run_uuid = str(uuid.uuid4())
    run_dir  = RUNS_DIR / "chess" / run_uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    request_dict = req.model_dump(exclude={"slow_mode"})

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,  # Use default executor (not the pipeline executor)
            lambda: analyse_chess_move(request_dict, str(run_dir), req.slow_mode),
        )
    except Exception as exc:
        # Chess analysis is a synchronous request/response call from the
        # Bluetooth bridge / Swift apps — there's no chat thread or
        # LangGraph checkpoint here to pause and resume the way the main
        # pipeline's retry-truncated flow does (see
        # pipeline/graph.py's _wrap_node_for_truncation_retry), so a
        # TruncatedOutputError here just becomes a clear 502 rather than
        # an opaque unhandled-exception 500. The Swift side can retry the
        # whole request (it already has the position) if it wants to.
        from clients.llm import TruncatedOutputError
        if isinstance(exc, TruncatedOutputError):
            raise HTTPException(
                status_code=502,
                detail=f"Chess analysis truncated at max_tokens={exc.cap} "
                       f"({exc.tokens_out} tokens generated) — try again, "
                       f"or use fast mode if this was slow mode.",
            )
        raise
    return result

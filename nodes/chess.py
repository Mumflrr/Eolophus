"""
nodes/chess.py — chess position analysis node.

Two sub-modes matching LLMHookService.swift:
  fast — every move, short thinking budget, no NoWait suppression
  slow — flagged moves (blunders, mirages, sacrifices), full thinking

Input:  ChessCoachingRequest (dict matching Swift struct)
Output: ChessCoachingOutput  (dict with headline/explanation/suggestion/tacticalPattern)
        + internal_reasoning field the Swift UI ignores but which improves quality

Key design decisions vs the Swift implementation:
  - Board state passed as coordinate list (NOT FEN — LLMs misread FEN)
  - internal_reasoning forces chain-of-thought before final fields
  - NoWait suppression disabled — chess reasoning needs backtracking
  - Thinking mode ON for both fast and slow (budget differs)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from clients.llm import call_model, get_model_config
from pipeline.state import PipelineState

log = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────────────
# Mirrors ChessCoachingOutput in Swift but adds internal_reasoning.
# Swift side ignores internal_reasoning — it exists to force CoT.

from pydantic import BaseModel, Field

class ChessAnalysisOutput(BaseModel):
    internal_reasoning: str = Field(
        description=(
            "3-4 sentences of raw chess logic. Analyze: where are the pieces, "
            "what does the engine line achieve, what weakness does this move create "
            "or exploit, why is the best alternative better. "
            "Swift UI ignores this field — write freely."
        )
    )
    headline: str = Field(
        description="One sentence: what happened and its immediate consequence."
    )
    explanation: str = Field(
        description=(
            "One sentence (two for slow mode complex moves): "
            "the concrete tactical or positional reason. Name pieces and squares."
        )
    )
    suggestion: Optional[str] = Field(
        default=None,
        description=(
            "One sentence: what to play instead and why. "
            "OMIT entirely for Excellent or Good moves."
        )
    )
    tactical_pattern: str = Field(
        description=(
            "One of: fork, pin, skewer, discovered_attack, back_rank, king_safety, "
            "development, pawn_structure, material_gain, zugzwang, passed_pawn, "
            "sacrifice, blunder, best_move, other"
        ),
        alias="tacticalPattern"
    )

    model_config = {"populate_by_name": True}


# ── System prompts ────────────────────────────────────────────────────────────

# Import improved prompts with CoT exemplars and calibration anchors
from nodes.chess_prompts import SYSTEM_FAST as _SYSTEM_FAST, SYSTEM_SLOW as _SYSTEM_SLOW, FEW_SHOT_EXAMPLES as _FEW_SHOT_EXAMPLES


def _format_board(request: dict) -> str:
    """
    Format board state as a coordinate list rather than FEN.
    LLMs understand 'White: King e1, Rook h1' far better than FEN strings.
    If no piece list is provided, falls back to omitting the section.
    """
    pieces = request.get("pieces")  # optional dict: {"white": [...], "black": [...]}
    if not pieces:
        return ""

    lines = ["Board position:"]
    for side in ("white", "black"):
        side_pieces = pieces.get(side, [])
        if side_pieces:
            lines.append(f"  {side.capitalize()}: {', '.join(side_pieces)}")
    return "\n".join(lines) + "\n"


def _build_prompt(request: dict, slow: bool) -> str:
    """Build the chess coaching prompt from a ChessCoachingRequest dict."""
    lines = []

    # Move identity
    move    = request.get("movePlayed", "?")
    side    = request.get("side", "?").capitalize()
    notation= request.get("moveNotation", "")
    quality = request.get("classification", "?")
    cp_loss = request.get("cpLoss")

    cp_str = ""
    if cp_loss and cp_loss > 0:
        cp_str = f" (−{cp_loss/100:.2f} pawns vs best)"

    lines.append(f"Move: {side} {notation} {move} — {quality}{cp_str}")

    # Best alternative
    best      = request.get("bestMove")
    best_eval = request.get("bestMoveEval")
    if best and best != move:
        ev = f" ({best_eval:+.2f})" if best_eval is not None else ""
        lines.append(f"Best was: {best}{ev}")

    # Position
    eval_after = request.get("evalAfter")
    if eval_after is not None:
        favour = ("white favoured" if eval_after > 0.2
                  else "black favoured" if eval_after < -0.2
                  else "roughly equal")
        phase = request.get("gamePhase", "")
        phase_str = f" | {phase}" if phase else ""
        lines.append(f"Eval: {eval_after:+.2f} ({favour}{phase_str})")

    w, d, b = request.get("winPctWhite"), request.get("winPctDraw"), request.get("winPctBlack")
    if w is not None and d is not None and b is not None:
        lines.append(f"Win odds: White {w}% / Draw {d}% / Black {b}%")

    mat = request.get("materialDelta", 0)
    if mat != 0:
        ahead = "White" if mat > 0 else "Black"
        lines.append(f"{ahead} up {abs(mat)} pawn(s) material")

    # Depth profile
    dp = request.get("depthProfile")
    if dp == "mirage":
        lines.append("⚠️ Score collapses at deeper search — hidden refutation exists")
    elif dp == "deepening":
        lines.append("✓ Score improves at depth — a forcing sequence is available")
    elif dp == "sharp":
        lines.append("⚡ Sharp — score oscillates, both sides have resources")

    # Pre-digested tactical flags
    flags = request.get("tacticalFlags", [])
    if flags:
        lines.append("Flags: " + " | ".join(flags))

    # Engine line
    best_line = request.get("bestLine", [])
    if best_line:
        lines.append("Engine line: " + " ".join(best_line[:4]))

    # Board position (coordinate list format — not FEN)
    board_str = _format_board(request)
    if board_str:
        lines.append(board_str)

    # Few-shot CoT examples — prime the model before the actual position
    if slow:
        lines.append(_FEW_SHOT_EXAMPLES)

    # Output schema — internal_reasoning first forces CoT
    schema_note = "explanation can be two sentences for complex moves" if slow else "one sentence per field"
    lines.append(f"""
Respond ONLY with valid JSON ({schema_note}, no markdown):
{{
  "internal_reasoning": "<3-4 sentences: analyze pieces, squares, engine line logic, why best move is better>",
  "headline": "<what happened and immediate consequence>",
  "explanation": "<specific tactical/positional reason — name pieces and squares>",
  "suggestion": "<what to play instead and why — OMIT this key entirely if Excellent or Good>",
  "tacticalPattern": "<fork|pin|skewer|discovered_attack|back_rank|king_safety|development|pawn_structure|material_gain|zugzwang|passed_pawn|sacrifice|blunder|best_move|other>"
}}""")

    return "\n".join(lines)


def analyse_chess_move(
    request:    dict,
    run_dir:    str,
    slow_mode:  bool = False,
) -> dict:
    """
    Analyse a chess move and return ChessCoachingOutput-compatible dict.

    Args:
        request:   ChessCoachingRequest as dict (from Swift JSON body)
        run_dir:   Run directory for logging
        slow_mode: True for flagged moves (blunders, mirages, sacrifices)

    Returns:
        dict with headline, explanation, suggestion, tacticalPattern keys
        (internal_reasoning stripped — Swift doesn't use it)
    """
    system  = _SYSTEM_SLOW if slow_mode else _SYSTEM_FAST
    budget  = 1024 if slow_mode else 512
    role    = "chess_slow" if slow_mode else "chess_fast"

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": _build_prompt(request, slow_mode)},
    ]

    result: ChessAnalysisOutput = call_model(
        model_id        = "9b",
        messages        = messages,
        response_schema = ChessAnalysisOutput,
        stage           = "chess_slow" if slow_mode else "chess_fast",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = budget,
        skip_nowait     = True,   # chess reasoning needs backtracking tokens
        max_retries     = 0,
    )

    log.info(
        "Chess analysis (%s): pattern=%s | has_suggestion=%s",
        "slow" if slow_mode else "fast",
        result.tactical_pattern,
        result.suggestion is not None,
    )

    # Return in ChessCoachingOutput format (drop internal_reasoning)
    output = {
        "headline":       result.headline,
        "explanation":    result.explanation,
        "tacticalPattern":result.tactical_pattern,
    }
    if result.suggestion:
        output["suggestion"] = result.suggestion

    return output

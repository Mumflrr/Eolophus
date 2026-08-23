"""
nodes/chess.py — chess position analysis node.

Two sub-modes matching LLMHookService.swift:
  fast — every move, short thinking budget, no NoWait suppression
  slow — flagged moves (blunders, mirages, sacrifices), full thinking

Both modes use the 9B model. The 35B would require a VRAM swap from the
always-hot 9B, adding unacceptable latency to a real-time chess UI.
Thinking budgets are set in routing.yaml (chess_fast: 512, chess_slow: 1024).

Prompt content lives in config/prompts/chess_fast.yaml and chess_slow.yaml.
The few-shot examples from chess_prompts.py are now inline in chess_slow.yaml.

Key design decisions:
  - Board state passed as coordinate list (NOT FEN — LLMs misread FEN)
  - internal_reasoning forces chain-of-thought before final fields
  - NoWait suppression disabled — chess reasoning needs backtracking
  - Thinking mode ON for both fast and slow (budget differs)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from clients.llm import call_role
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────
# Mirrors ChessCoachingOutput in Swift but adds internal_reasoning.
# Swift side ignores internal_reasoning — it exists to force CoT.

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


# ── Board context builder ─────────────────────────────────────────────────────

def _build_board_context(request: dict, slow: bool) -> str:
    """
    Build the full user message content from a ChessCoachingRequest dict.
    Passed as the {board_context} template variable to chess_fast/slow.yaml.
    Board state is expressed as coordinate list (NOT FEN — LLMs misread FEN).
    """
    lines = []

    # Move identity
    move     = request.get("movePlayed", "?")
    side     = request.get("side", "?").capitalize()
    notation = request.get("moveNotation", "")
    quality  = request.get("classification", "?")
    cp_loss  = request.get("cpLoss")

    cp_str = f" (−{cp_loss/100:.2f} pawns vs best)" if cp_loss and cp_loss > 0 else ""
    lines.append(f"Move: {side} {notation} {move} — {quality}{cp_str}")

    # Best alternative
    best      = request.get("bestMove")
    best_eval = request.get("bestMoveEval")
    if best and best != move:
        ev = f" ({best_eval:+.2f})" if best_eval is not None else ""
        lines.append(f"Best was: {best}{ev}")

    # Evaluation
    eval_after = request.get("evalAfter")
    if eval_after is not None:
        favour = (
            "white favoured" if eval_after > 0.2
            else "black favoured" if eval_after < -0.2
            else "roughly equal"
        )
        phase     = request.get("gamePhase", "")
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

    # Tactical flags
    flags = request.get("tacticalFlags", [])
    if flags:
        lines.append("Flags: " + " | ".join(flags))

    # Engine line
    best_line = request.get("bestLine", [])
    if best_line:
        lines.append("Engine line: " + " ".join(best_line[:4]))

    # Board position as coordinate list
    pieces = request.get("pieces")
    if pieces:
        lines.append("Board position:")
        for side_name in ("white", "black"):
            side_pieces = pieces.get(side_name, [])
            if side_pieces:
                lines.append(f"  {side_name.capitalize()}: {', '.join(side_pieces)}")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_chess_move(
    request:   dict,
    run_dir:   str,
    slow_mode: bool = False,
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
    role          = "chess_slow" if slow_mode else "chess_fast"
    board_context = _build_board_context(request, slow_mode)

    result: ChessAnalysisOutput = call_role(
        role            = role,
        template_vars   = {"board_context": board_context},
        response_schema = ChessAnalysisOutput,
        stage           = role,
        run_dir         = run_dir,
        thinking        = True,
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
        "headline":        result.headline,
        "explanation":     result.explanation,
        "tacticalPattern": result.tactical_pattern,
    }
    if result.suggestion:
        output["suggestion"] = result.suggestion

    return output

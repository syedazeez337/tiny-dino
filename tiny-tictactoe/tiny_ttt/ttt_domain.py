"""Full reachable-position domain for tic-tac-toe certification."""
from __future__ import annotations

import hashlib
import json

from .ttt_engine import EMPTY_BOARD, GameState, X, apply_move, legal_cells

DOMAIN_VERSION = "ttt-domain-v1"


def enumerate_domain() -> tuple[GameState, ...]:
    """Every reachable position, deduped by board, sorted deterministically.

    A board uniquely determines the player to move (X moves first), so keying on
    the board is sufficient. Includes terminal positions.
    """
    found: dict[tuple[int, ...], GameState] = {}
    stack: list[GameState] = [GameState(EMPTY_BOARD, X, 0)]
    while stack:
        state = stack.pop()
        if state.board in found:
            continue
        found[state.board] = state
        if state.is_terminal:
            continue
        for cell in legal_cells(state.board):
            stack.append(apply_move(state, cell))
    return tuple(sorted(found.values(), key=lambda state: state.board))


def decision_states() -> tuple[GameState, ...]:
    """Domain positions with at least one legal move (the policy's decisions)."""
    return tuple(state for state in enumerate_domain() if not state.is_terminal)


def domain_manifest() -> dict[str, object]:
    states = enumerate_domain()
    canonical = json.dumps(
        [{"board": list(state.board), "player": state.player, "winner": state.winner} for state in states],
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "version": DOMAIN_VERSION,
        "positions": len(states),
        "decisions": len(decision_states()),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }

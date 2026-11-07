"""Exact minimax teacher for tic-tac-toe. Never imported by live play."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .ttt_engine import DRAW, Board, GameState, apply_move, board_winner, legal_cells, other

WIN = 1
LOSS = -1
DRAW_VALUE = 0


@lru_cache(maxsize=None)
def _value(board: Board, player: int) -> int:
    """Game-theoretic outcome for the player to move: +1 win, 0 draw, -1 loss.

    Terminal positions are only reached right after a move, so if a winner exists
    it is the opponent of ``player``; the value for ``player`` is therefore -1.
    """
    winner = board_winner(board)
    if winner == DRAW:
        return DRAW_VALUE
    if winner != 0:
        return LOSS
    best = -2
    for cell in legal_cells(board):
        next_board = list(board)
        next_board[cell] = player
        child = tuple(next_board)  # type: ignore[assignment]
        value = -_value(child, other(player))
        if value > best:
            best = value
    return best


@dataclass(frozen=True, slots=True)
class TeacherVerdict:
    acceptable: tuple[int, ...]  # optimal legal cells, ascending
    best_outcome: int  # WIN / DRAW_VALUE / LOSS for the player to move
    outcomes: dict[int, int]  # per-legal-cell outcome for the player to move
    horizon_frames: int  # always the full game (exact), kept for parity with Dino

    def acceptable_mask(self) -> tuple[bool, ...]:
        return tuple(cell in self.acceptable for cell in range(9))


class Teacher:
    def evaluate(self, state: GameState) -> TeacherVerdict:
        if state.is_terminal:
            raise ValueError("cannot evaluate a terminal position")
        outcomes: dict[int, int] = {}
        for cell in legal_cells(state.board):
            child = apply_move(state, cell)
            outcomes[cell] = -_value(child.board, child.player)
        best = max(outcomes.values())
        acceptable = tuple(sorted(cell for cell, value in outcomes.items() if value == best))
        return TeacherVerdict(acceptable, best, outcomes, horizon_frames=9)


def optimal_value(state: GameState) -> int:
    """Outcome for the player to move with perfect play."""
    if state.is_terminal:
        return DRAW_VALUE
    return _value(state.board, state.player)

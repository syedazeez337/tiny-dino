"""Deterministic 3x3 tic-tac-toe engine.

The engine is the single source of truth for board rules, legal moves, and
game outcomes. Everything else (teacher, features, policy, certification)
reads states produced here.
"""
from __future__ import annotations

from dataclasses import dataclass

EMPTY = 0
X = 1
O = 2
DRAW = 3

LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

Board = tuple[int, int, int, int, int, int, int, int, int]

EMPTY_BOARD: Board = (EMPTY,) * 9


def other(player: int) -> int:
    """The opposing player."""
    if player not in (X, O):
        raise ValueError(f"unknown player: {player}")
    return O if player == X else X


def board_winner(board: Board) -> int:
    """Return X, O, DRAW, or 0 (game still in progress)."""
    for a, b, c in LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell != EMPTY for cell in board):
        return DRAW
    return 0


def legal_cells(board: Board) -> tuple[int, ...]:
    return tuple(index for index, cell in enumerate(board) if cell == EMPTY)


@dataclass(frozen=True, slots=True)
class GameState:
    board: Board
    player: int  # the player to move
    winner: int  # 0 none, X, O, or DRAW

    @property
    def is_terminal(self) -> bool:
        return self.winner != 0


class TicTacToeEngine:
    def __init__(self) -> None:
        self._state: GameState | None = None

    def reset(self) -> None:
        self._state = GameState(EMPTY_BOARD, X, 0)

    def snapshot(self) -> GameState:
        if self._state is None:
            raise RuntimeError("engine has not been reset")
        return self._state

    def restore(self, state: GameState) -> None:
        self._state = state

    def legal_mask(self) -> tuple[bool, ...]:
        return tuple(cell == EMPTY for cell in self.snapshot().board)

    def is_terminal(self) -> bool:
        return self.snapshot().is_terminal

    def step(self, cell: int) -> None:
        state = self.snapshot()
        if state.winner != 0:
            raise RuntimeError("cannot step a terminal game")
        if not 0 <= cell < 9:
            raise ValueError(f"cell out of range: {cell}")
        if state.board[cell] != EMPTY:
            raise ValueError(f"cell already occupied: {cell}")
        board = list(state.board)
        board[cell] = state.player
        next_board = tuple(board)  # type: ignore[assignment]
        winner = board_winner(next_board)
        self._state = GameState(next_board, other(state.player), winner)

    def clone(self) -> TicTacToeEngine:
        clone = TicTacToeEngine()
        clone.restore(self.snapshot())
        return clone


def apply_move(state: GameState, cell: int) -> GameState:
    """Pure functional move application (used by the teacher and the domain)."""
    if state.winner != 0:
        raise RuntimeError("cannot move in a terminal position")
    if state.board[cell] != EMPTY:
        raise ValueError(f"cell already occupied: {cell}")
    board = list(state.board)
    board[cell] = state.player
    next_board = tuple(board)  # type: ignore[assignment]
    return GameState(next_board, other(state.player), board_winner(next_board))

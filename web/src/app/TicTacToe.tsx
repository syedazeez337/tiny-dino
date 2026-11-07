import { useCallback, useEffect, useState } from 'react'
import { formatBytes } from '../lib/formatting'
import {
  fetchTttModel,
  playMove,
  startGame,
  type TttModel,
  type TttState,
} from '../lib/ttt'

const MARK: Record<number, string> = { 1: 'X', 2: 'O' }

function sideOf(value: number): string | null {
  if (value === 1) return 'x'
  if (value === 2) return 'o'
  return null
}

const STATUS: Record<string, string> = {
  playing: 'Playing',
  human_win: 'You win',
  model_win: 'Model wins',
  draw: 'Draw',
}

const BANNER: Record<string, string> = {
  playing: '',
  human_win: 'You win — unexpected, the model is certified never to lose',
  model_win: 'The model wins',
  draw: 'Draw — the model never loses',
}

export function TicTacToe() {
  const [model, setModel] = useState<TttModel | null>(null)
  const [state, setState] = useState<TttState | null>(null)
  const [human, setHuman] = useState('x')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void fetchTttModel()
      .then(setModel)
      .catch((caught) => setError(caught instanceof Error ? caught.message : 'model unavailable'))
    void startGame('x')
      .then(setState)
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : 'could not start a game'),
      )
  }, [])

  const newGame = useCallback((side: string) => {
    setHuman(side)
    setError(null)
    void startGame(side)
      .then(setState)
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : 'could not start a game'),
      )
  }, [])

  const choose = useCallback(
    (cell: number) => {
      if (!state) return
      void playMove(state.run_id, cell)
        .then(setState)
        .catch((caught) => setError(caught instanceof Error ? caught.message : 'move failed'))
    },
    [state],
  )

  if (error) {
    return (
      <div className="board board--message">
        <div className="banner banner--danger" role="alert">
          {error}
        </div>
      </div>
    )
  }
  if (!state) {
    return (
      <div className="board board--message">
        <div className="skeleton" />
      </div>
    )
  }

  const myTurn = state.status === 'playing' && state.turn === state.human
  const probabilities = state.probabilities ?? []
  const cert = model?.certification

  return (
    <div className="board">
      <header className="board__bar">
        <span className="board__chip" />
        <div className="board__titles">
          <span className="board__name">Tic-Tac-Toe</span>
          <span className="board__detail">
            Unassisted neural policy
            {model
              ? ` · ${model.params.toLocaleString()} parameters · ${model.architecture.hidden}×${model.architecture.layers}`
              : ''}
            {cert ? ` · certified perfect play over ${cert.positions.toLocaleString()} positions` : ''}
          </span>
        </div>
        <span className="board__status">{STATUS[state.status] ?? state.status}</span>
        <div className="board__actions">
          <label className="field">
            you play
            <select value={human} onChange={(event) => newGame(event.target.value)}>
              <option value="x">X (first)</option>
              <option value="o">O (second)</option>
            </select>
          </label>
          <button className="button button--secondary" onClick={() => newGame(human)}>
            New game
          </button>
        </div>
      </header>

      <main className="board__main">
        <section className="board__lane">
          <div className="ttt-grid" aria-label="tic-tac-toe board">
            {state.board.map((value, index) => {
              const side = sideOf(value)
              const classes = [
                'ttt-cell',
                side ? `ttt-cell--${side}` : '',
                index === state.model_move ? 'ttt-cell--last' : '',
              ]
                .filter(Boolean)
                .join(' ')
              return (
                <button
                  key={index}
                  className={classes}
                  disabled={Boolean(value) || !myTurn}
                  onClick={() => choose(index)}
                  aria-label={`cell ${index + 1}${value ? `, ${MARK[value] ?? ''}` : ''}`}
                >
                  {value ? (MARK[value] ?? '') : index + 1}
                </button>
              )
            })}
          </div>
          <p className="lane__pipeline">
            {state.status === 'playing' ? (myTurn ? 'YOUR MOVE' : 'MODEL THINKING') : ''}
          </p>
          {BANNER[state.status] ? <div className="lane__replay">{BANNER[state.status]}</div> : null}
        </section>

        <aside className="board__side">
          <div className="probs" aria-label="Move probabilities">
            {Array.from({ length: 9 }, (_, index) => {
              const value = probabilities[index] ?? 0
              const chosen = state.model_move === index
              return (
                <div key={index} className={chosen ? 'prob prob--chosen' : 'prob'}>
                  <span>{index + 1}</span>
                  <span className="prob__track">
                    <span className="prob__fill" style={{ width: `${Math.round(value * 100)}%` }} />
                  </span>
                  <span className="prob__value">{value.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
          <div className="tiles">
            <div className="tile">
              <span className="tile__label">Model move</span>
              <span className="tile__value">
                {state.model_move === null ? '—' : state.model_move + 1}
              </span>
            </div>
            <div className="tile">
              <span className="tile__label">Latency</span>
              <span className="tile__value">
                {state.latency_ms === null ? '—' : `${state.latency_ms.toFixed(2)} ms`}
              </span>
            </div>
            <div className="tile">
              <span className="tile__label">Result</span>
              <span className="tile__value">{STATUS[state.status] ?? '—'}</span>
            </div>
            <div className="tile">
              <span className="tile__label">You are</span>
              <span className="tile__value">{state.human.toUpperCase()}</span>
            </div>
          </div>
        </aside>
      </main>

      <footer className="board__foot">
        <dl className="facts">
          <div>
            <dt>model</dt>
            <dd>{model?.name ?? 'Tiny Tic-Tac-Toe Policy'}</dd>
          </div>
          <div>
            <dt>parameters</dt>
            <dd>{model ? model.params.toLocaleString() : '—'}</dd>
          </div>
          <div>
            <dt>checkpoint</dt>
            <dd>{model ? formatBytes(model.bytes) : '—'}</dd>
          </div>
          <div>
            <dt>features</dt>
            <dd>{model?.feature_version ?? '—'}</dd>
          </div>
          <div>
            <dt>certification</dt>
            <dd>{cert ? (cert.perfect_play ? 'perfect play' : 'uncertified') : '—'}</dd>
          </div>
          <div>
            <dt>interventions</dt>
            <dd>0</dd>
          </div>
          <div className="facts__wide">
            <dt>sha-256</dt>
            <dd>{model?.checkpoint_sha256 ?? '—'}</dd>
          </div>
          <div className="facts__wide">
            <dt>runtime</dt>
            <dd>{cert?.claim ?? 'greedy argmax executed unchanged · no planner, no shield'}</dd>
          </div>
        </dl>
      </footer>
    </div>
  )
}

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../app/App'

const DINO_MODEL = {
  name: 'Tiny Dino Policy',
  params: 21699,
  checkpoint_bytes: 90929,
  checkpoint_sha256: 'b4cd918bb5fae7d0a826ef23a055cbe05e70505f3487243e43d0919813f11109',
  device: 'cpu',
  engine_revision: 'dino-engine-v2',
  feature_version: 'dino-features-v1',
  runtime_interventions: 0,
}

const TTT_MODEL = {
  name: 'TinyTicTacToe',
  architecture: { kind: 'mlp', hidden: 96, layers: 3 },
  params: 22281,
  bytes: 93233,
  checkpoint_sha256: 'e5f173bacb9f1adf1ec76d15f623d6f2d153d05aac97defc9c1aceb4f48ed875',
  feature_version: 'ttt-features-v1',
  certification: {
    positions: 5478,
    decisions: 4520,
    optimal: 4520,
    non_optimal: 0,
    illegal: 0,
    perfect_play: true,
    claim: 'perfect play',
  },
}

const TTT_START = {
  run_id: 'run-1',
  board: [0, 0, 0, 0, 0, 0, 0, 0, 0],
  turn: 'x',
  human: 'x',
  status: 'playing',
  model_move: null,
  probabilities: null,
  latency_ms: null,
}

const TTT_AFTER = {
  run_id: 'run-1',
  board: [2, 0, 0, 0, 1, 0, 0, 0, 0],
  turn: 'x',
  human: 'x',
  status: 'playing',
  model_move: 0,
  probabilities: [0.9, 0.01, 0, 0, 0.02, 0, 0, 0, 0.07],
  latency_ms: 0.5,
}

function stubFetch() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      const body = url.includes('/api/ttt/model')
        ? TTT_MODEL
        : url.includes('/api/ttt/play/start')
          ? TTT_START
          : url.includes('/api/ttt/play/move')
            ? TTT_AFTER
            : DINO_MODEL
      return { ok: true, status: 200, json: async () => body }
    }) as unknown as typeof fetch,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('Tiny Games shell', () => {
  it('shows the tab bar and switches to tic-tac-toe', async () => {
    stubFetch()
    const user = userEvent.setup()
    render(<App />)

    expect(screen.getByText('Tiny Games')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Dino' })).toHaveAttribute('aria-selected', 'true')

    await user.click(screen.getByRole('tab', { name: 'Tic-Tac-Toe' }))

    const board = await screen.findByLabelText('tic-tac-toe board')
    expect(within(board).getAllByRole('button')).toHaveLength(9)
  })

  it('posts a move and shows the model reply', async () => {
    stubFetch()
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('tab', { name: 'Tic-Tac-Toe' }))
    const board = await screen.findByLabelText('tic-tac-toe board')

    await user.click(within(board).getByRole('button', { name: /cell 5/ }))

    const updated = await screen.findByLabelText('tic-tac-toe board')
    expect(within(updated).getByRole('button', { name: /cell 1, O/ })).toBeInTheDocument()
  })
})

import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../app/App'

const MODEL = {
  name: 'Tiny Dino Policy',
  params: 21699,
  checkpoint_bytes: 90929,
  checkpoint_sha256: 'b4cd918bb5fae7d0a826ef23a055cbe05e70505f3487243e43d0919813f11109',
  device: 'cuda',
  engine_revision: 'dino-engine-v2',
  feature_version: 'dino-features-v1',
  runtime_interventions: 0,
}

afterEach(() => vi.unstubAllGlobals())

describe('Tiny Dino board', () => {
  it('fills the page with the lane, live probabilities and metrics', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, status: 200, json: async () => MODEL })) as unknown as typeof fetch,
    )
    render(<App />)

    const probs = await screen.findByLabelText('Next move probabilities')
    expect(within(probs).getByText('RUN')).toBeInTheDocument()
    expect(within(probs).getByText('JUMP ↑')).toBeInTheDocument()
    expect(within(probs).getByText('DUCK ↓')).toBeInTheDocument()

    expect(screen.getByRole('img', { name: 'Dino lane' })).toBeInTheDocument()
    expect(screen.getByText('SCORE OVER TIME')).toBeInTheDocument()
    expect(screen.getByText('Answer time')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start run' })).toBeInTheDocument()
    expect(screen.getByText(/unassisted neural policy/i)).toBeInTheDocument()

    // The page is the game: no destination tabs remain.
    expect(screen.queryByRole('button', { name: 'Proof' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Arena' })).toBeNull()
  })

  it('surfaces checkpoint errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 503,
        json: async () => ({ detail: 'checkpoint missing' }),
      })) as unknown as typeof fetch,
    )
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('checkpoint missing')
  })
})

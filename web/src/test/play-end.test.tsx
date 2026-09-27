import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../app/App'

const MODEL = {
  name: 'Tiny Dino Policy',
  params: 21699,
  checkpoint_bytes: 90929,
  checkpoint_sha256: 'b4cd918bb5fae7d0a826ef23a055cbe05e70505f3487243e43d0919813f11109',
  device: 'cpu',
  engine_revision: 'dino-engine-v2',
  runtime_interventions: 0,
}

const state = (frame: number) => ({
  frame,
  distance: frame * 6,
  speed: 360,
  dino_y: 0,
  dino_vy: 0,
  ducking: false,
  control: 'run',
  motion: 'grounded',
  obstacle: null,
  next_obstacle: null,
})

const frames = [1, 2, 3].map((frame) => ({
  action: 'run',
  probabilities: [1, 0, 0],
  latency_ms: 0.1,
  state: state(frame),
}))

/** Serve one three-frame chunk that ends the session with the given status. */
function stubServer(endStatus: 'crashed' | 'complete') {
  const json = (body: unknown) => ({ ok: true, status: 200, json: async () => body })
  vi.stubGlobal(
    'fetch',
    vi.fn(async (path: string) => {
      if (path === '/api/model') return json(MODEL)
      if (path === '/api/play/start') return json({ session_id: 's1', state: state(0) })
      if (path === '/api/play/step') {
        return json({ session_id: 's1', state: state(3), frames, status: endStatus })
      }
      return json({
        frames: frames.map((item, index) => ({ index: index + 1, action: 'run', state: item.state })),
      })
    }) as unknown as typeof fetch,
  )
  // jsdom has no animation loop of its own; drive the game clock with timers.
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
    window.setTimeout(() => callback(performance.now()), 16),
  )
  vi.stubGlobal('cancelAnimationFrame', (handle: number) => window.clearTimeout(handle))
}

const deaths = () => within(screen.getByText('Deaths').parentElement as HTMLElement)

afterEach(() => vi.unstubAllGlobals())

describe('end of a run', () => {
  it('counts a crash as a death and labels the replay as a crash', async () => {
    stubServer('crashed')
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: 'Start run' }))

    expect(await screen.findByText('Crashed')).toBeInTheDocument()
    expect(deaths().getByText('1')).toBeInTheDocument()
    expect(await screen.findByText(/CRASH REPLAY/)).toBeInTheDocument()
  })

  it('reports the session limit as complete, not as a crash', async () => {
    stubServer('complete')
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: 'Start run' }))

    expect(await screen.findByText('Complete')).toBeInTheDocument()
    expect(screen.getByText(/reached the session limit/)).toBeInTheDocument()
    expect(screen.queryByText('Crashed')).toBeNull()
    expect(deaths().getByText('0')).toBeInTheDocument()
  })
})

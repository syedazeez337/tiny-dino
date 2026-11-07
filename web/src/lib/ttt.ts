export interface TttModel {
  name: string
  architecture: { kind: string; hidden: number; layers: number }
  params: number
  bytes: number
  checkpoint_sha256: string
  feature_version: string
  certification?: {
    positions: number
    decisions: number
    optimal: number
    non_optimal: number
    illegal: number
    perfect_play: boolean
    claim: string
  }
}

export interface TttState {
  run_id: string
  board: number[]
  turn: string | null
  human: string
  status: string
  model_move: number | null
  probabilities: number[] | null
  latency_ms: number | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { 'content-type': 'application/json' }, ...init })
  const body = (await response.json()) as T & { detail?: string }
  if (!response.ok) throw new Error(body.detail ?? `Request failed (${response.status})`)
  return body
}

export function fetchTttModel(): Promise<TttModel> {
  return request<TttModel>('/api/ttt/model')
}

export function startGame(human: string): Promise<TttState> {
  return request<TttState>('/api/ttt/play/start', {
    method: 'POST',
    body: JSON.stringify({ human }),
  })
}

export function playMove(runId: string, cell: number): Promise<TttState> {
  return request<TttState>('/api/ttt/play/move', {
    method: 'POST',
    body: JSON.stringify({ run_id: runId, cell }),
  })
}

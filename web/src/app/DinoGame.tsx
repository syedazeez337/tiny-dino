import { useCallback, useEffect, useRef, useState } from 'react'
import { formatBytes } from '../lib/formatting'
import { GameCanvas } from './GameCanvas'
import { ProbabilityPanel } from './ProbabilityPanel'
import { Bars, LineChart } from './Charts'
import {
  argmax,
  bucketMedians,
  median,
  scoreFromDistance,
  type EngineState,
  type Frame,
} from './arena'

type Model = {
  name?: string
  params: number
  checkpoint_bytes: number
  checkpoint_sha256: string
  device?: string
  engine_revision: string
  feature_version?: string
  runtime_interventions?: number
}

interface StepResponse {
  session_id: string
  state: EngineState
  frames: Frame[]
  status: string
}

interface ReplayResponse {
  frames: { index: number; action: string; state: EngineState }[]
}

const CHUNK = 30
const FRAME_MS = 1000 / 60
const REPLAY_FRAMES = 120

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { 'content-type': 'application/json' }, ...init })
  const body = (await response.json()) as T & { detail?: string }
  if (!response.ok) throw new Error(body.detail ?? `Request failed (${response.status})`)
  return body
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="tile">
      <span className="tile__label">{label}</span>
      <span className="tile__value">{value}</span>
      <span className="tile__sub">{sub ?? ''}</span>
    </div>
  )
}

const STATUS: Record<string, string> = {
  idle: 'Idle',
  running: 'Running',
  paused: 'Paused',
  stopped: 'Stopped',
  crashed: 'Crashed',
  complete: 'Complete',
  error: 'Error',
}

function usePlay() {
  const [seed, setSeed] = useState(201)
  const [session, setSession] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [display, setDisplay] = useState<Frame | null>(null)
  const [tick, setTick] = useState(0)
  const [crashed, setCrashed] = useState(false)
  const [completed, setCompleted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scoreSeries, setScoreSeries] = useState<{ t: number; v: number }[]>([])
  const [latencySeries, setLatencySeries] = useState<{ t: number; v: number }[]>([])
  const [replayFrame, setReplayFrame] = useState<Frame | null>(null)
  const [frames, setFrames] = useState(0)
  const [agreement, setAgreement] = useState({ agree: 0, checked: 0 })
  const [diverged, setDiverged] = useState<number | null>(null)
  const [stopped, setStopped] = useState(false)

  const buffer = useRef<Frame[]>([])
  const pending = useRef(false)
  const exhausted = useRef(false)
  const endStatus = useRef<string | null>(null)
  const finished = useRef(false)
  const elapsed = useRef(0)
  const previous = useRef<number | null>(null)
  const tickRef = useRef(0)
  const framesRef = useRef(0)
  const scoreRef = useRef<{ t: number; v: number }[]>([])
  const latencyRef = useRef<{ t: number; v: number }[]>([])
  const bestRef = useRef(0)
  const agreeRef = useRef(0)
  const checkedRef = useRef(0)
  const divergedRef = useRef<number | null>(null)
  const runningRef = useRef(false)
  const sessionRef = useRef<string | null>(null)
  const replayRef = useRef<Frame[] | null>(null)
  const replayProgress = useRef(0)

  useEffect(() => {
    runningRef.current = running
  }, [running])
  useEffect(() => {
    sessionRef.current = session
  }, [session])

  const fetchChunk = useCallback(async (id: string) => {
    if (pending.current) return
    pending.current = true
    try {
      const response = await request<StepResponse>('/api/play/step', {
        method: 'POST',
        body: JSON.stringify({ session_id: id, frames: CHUNK }),
      })
      if (sessionRef.current !== id) return
      buffer.current.push(...response.frames)
      if (response.status !== 'running') {
        exhausted.current = true
        endStatus.current = response.status
      }
    } catch (caught) {
      if (sessionRef.current === id) {
        setError(caught instanceof Error ? caught.message : 'Play request failed')
        runningRef.current = false
        setRunning(false)
      }
    } finally {
      pending.current = false
    }
  }, [])

  const loadReplay = useCallback(async () => {
    const id = sessionRef.current
    if (!id) return
    try {
      const start = Math.max(0, framesRef.current - REPLAY_FRAMES)
      const response = await request<ReplayResponse>(
        `/api/play/replay/${id}?start=${start}&limit=${REPLAY_FRAMES}`,
      )
      const clip = response.frames.map((item) => ({
        action: item.action,
        probabilities: [],
        latency_ms: 0,
        state: item.state,
      }))
      if (clip.length === 0) return
      replayRef.current = clip
      replayProgress.current = 0
      setReplayFrame(clip[0] ?? null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Replay request failed')
    }
  }, [])

  const stop = useCallback(() => {
    runningRef.current = false
    setRunning(false)
    buffer.current = []
    exhausted.current = true
    finished.current = true
    setStopped(true)
  }, [])

  const start = useCallback(async () => {
    setError(null)
    setCrashed(false)
    setCompleted(false)
    setStopped(false)
    setDiverged(null)
    setReplayFrame(null)
    buffer.current = []
    scoreRef.current = []
    latencyRef.current = []
    setScoreSeries([])
    setLatencySeries([])
    elapsed.current = 0
    previous.current = null
    tickRef.current = 0
    framesRef.current = 0
    bestRef.current = 0
    agreeRef.current = 0
    checkedRef.current = 0
    divergedRef.current = null
    exhausted.current = false
    endStatus.current = null
    finished.current = false
    pending.current = false
    replayRef.current = null
    setAgreement({ agree: 0, checked: 0 })
    setFrames(0)
    try {
      const response = await request<{ session_id: string; state: EngineState }>(
        '/api/play/start',
        { method: 'POST', body: JSON.stringify({ seed }) },
      )
      sessionRef.current = response.session_id
      setSession(response.session_id)
      setDisplay({ action: 'run', probabilities: [1, 0, 0], latency_ms: 0, state: response.state })
      runningRef.current = true
      setRunning(true)
      void fetchChunk(response.session_id)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not start a run')
    }
  }, [fetchChunk, seed])

  useEffect(() => {
    let handle = 0
    const step = (now: number) => {
      handle = window.requestAnimationFrame(step)
      const previousTick = previous.current
      previous.current = now

      if (replayRef.current) {
        replayProgress.current += 0.5
        const clip = replayRef.current
        const index = Math.floor(replayProgress.current)
        if (index >= clip.length - 1) {
          // Hand the canvas back to the live display, which already holds the final frame.
          setReplayFrame(null)
          replayRef.current = null
        } else {
          setReplayFrame(clip[index] ?? null)
        }
      }

      if (!runningRef.current || previousTick === null) return
      let delta = now - previousTick
      if (delta > 250) delta = FRAME_MS
      elapsed.current += delta
      let sampled = false
      while (elapsed.current >= FRAME_MS && buffer.current.length > 0) {
        elapsed.current -= FRAME_MS
        const frame = buffer.current.shift()
        if (!frame) break
        tickRef.current += 1
        framesRef.current += 1
        setDisplay(frame)
        setTick(tickRef.current)
        const chosen = argmax(frame.probabilities)
        checkedRef.current += 1
        if (chosen === frame.action) agreeRef.current += 1
        else if (divergedRef.current === null) divergedRef.current = frame.state.frame
        const score = scoreFromDistance(frame.state.distance)
        bestRef.current = Math.max(bestRef.current, score)
        if (framesRef.current % 15 === 0) {
          scoreRef.current.push({ t: framesRef.current / 60, v: score })
          sampled = true
        }
        latencyRef.current.push({ t: framesRef.current / 60, v: frame.latency_ms })
        if (latencyRef.current.length > 1200) latencyRef.current.shift()
      }
      if (sampled) {
        setScoreSeries([...scoreRef.current])
        setLatencySeries([...latencyRef.current])
        setFrames(framesRef.current)
        setAgreement({ agree: agreeRef.current, checked: checkedRef.current })
        if (divergedRef.current !== null) setDiverged(divergedRef.current)
      }
      if (!finished.current) {
        if (buffer.current.length === 0 && exhausted.current) {
          finished.current = true
          runningRef.current = false
          setRunning(false)
          // The server also ends a session at its frame cap; only a terminal engine state is a crash.
          if (endStatus.current === 'crashed') {
            setCrashed(true)
            void loadReplay()
          } else {
            setCompleted(true)
          }
        } else if (buffer.current.length < CHUNK / 2 && !exhausted.current && sessionRef.current) {
          void fetchChunk(sessionRef.current)
        }
      }
    }
    handle = window.requestAnimationFrame(step)
    return () => window.cancelAnimationFrame(handle)
  }, [fetchChunk, loadReplay])

  const state = display?.state ?? null
  const chosen = display ? argmax(display.probabilities) : null
  const p50 = median(latencyRef.current.map((sample) => sample.v))
  const p95 = median(latencyRef.current.slice(-120).map((sample) => sample.v))
  const rate = agreement.checked > 0 ? (100 * agreement.agree) / agreement.checked : 100

  return { seed, setSeed, session, running, setRunning, display, tick, crashed, completed, error, scoreSeries, latencySeries, replayFrame, frames, agreement, diverged, stopped, state, chosen, p50, p95, rate, start, stop, loadReplay, best: bestRef.current, framesRef }
}

export function DinoGame() {
  const [model, setModel] = useState<Model | null>(null)
  const [error, setError] = useState<string | null>(null)
  const play = usePlay()

  useEffect(() => {
    void request<Model>('/api/model')
      .then(setModel)
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : 'Could not reach local server'),
      )
  }, [])

  if (error || !model) {
    return (
      <div className="board board--message">
        {error ? (
          <div className="banner banner--danger" role="alert">
            {error}
          </div>
        ) : (
          <div className="skeleton" />
        )}
      </div>
    )
  }

  const status = play.error
    ? 'error'
    : play.crashed
      ? 'crashed'
      : play.completed
        ? 'complete'
        : play.stopped
        ? 'stopped'
        : play.running
          ? 'running'
          : play.session
            ? 'paused'
            : 'idle'

  return (
    <div className="board">
      <header className="board__bar">
        <span className="board__chip" />
        <div className="board__titles">
          <span className="board__name">Tiny Dino</span>
          <span className="board__detail">
            Unassisted neural policy · {model.params.toLocaleString()} parameters ·{' '}
            {model.device ?? 'cpu'} · {model.engine_revision}
          </span>
        </div>
        <span className="board__live">
          frame {play.frames.toLocaleString()} · {Math.round(play.state?.speed ?? 0)} px/s
        </span>
        <span className={`board__status board__status--${status}`}>{STATUS[status]}</span>
        <div className="board__actions">
          <label className="field">
            seed
            <input
              value={play.seed}
              type="number"
              onChange={(event) => play.setSeed(Number(event.target.value))}
            />
          </label>
          <button
            className="button button--secondary"
            onClick={() => play.setRunning((value) => !value)}
            disabled={!play.session || play.stopped || play.crashed || play.completed}
          >
            {play.running ? 'Pause' : 'Resume'}
          </button>
          <button
            className="button button--secondary"
            onClick={play.stop}
            disabled={!play.session || play.stopped || play.crashed || play.completed}
          >
            Stop
          </button>
          <button
            className="button button--ghost"
            onClick={() => void play.loadReplay()}
            disabled={!play.session}
          >
            Replay
          </button>
          <button className="button button--primary" onClick={() => void play.start()}>
            Start run
          </button>
        </div>
      </header>

      <main className="board__main">
        <section className="board__lane">
          <div className="lane__game">
            <GameCanvas frame={play.replayFrame ?? play.display} tick={play.tick} className="arena-canvas" />
          </div>
          <p className="lane__pipeline">
            THINKING 0 → ANSWER {(play.chosen ?? '—').toUpperCase()} → ACTION{' '}
            {(play.display?.action ?? '—').toUpperCase()} · skipped 0
          </p>
          {play.replayFrame ? (
            <div className="lane__replay">
              {play.crashed ? 'CRASH REPLAY' : 'REPLAY'} · ½× · SCORE {scoreFromDistance(play.replayFrame.state.distance)} —
              reconstructed and hash-verified server-side; the live race is unaffected
            </div>
          ) : (
            <div className="lane__charts">
              <LineChart
                title="SCORE OVER TIME"
                points={play.scoreSeries}
                format={(v) => v.toFixed(0)}
              />
              <LineChart
                title="ANSWER TIME PER MOVE, MS"
                points={bucketMedians(play.latencySeries, 0.5)}
                format={(v) => v.toFixed(0)}
              />
              <div className="chart-stack">
                <Bars
                  title="ANSWER TIME, MEDIAN"
                  rows={[{ label: 'THIS RUN', value: play.p50 }]}
                  format={(v) => `${v.toFixed(0)} ms`}
                  max={Math.max(1, Math.ceil(play.p50 ?? 1))}
                />
                <Bars
                  title="BEST MOVE PICKED"
                  rows={[{ label: 'AGREEMENT', value: play.rate }]}
                  format={(v) => `${v.toFixed(0)}%`}
                  max={100}
                />
              </div>
            </div>
          )}
        </section>

        <aside className="board__side">
          <ProbabilityPanel
            probabilities={play.display?.probabilities ?? []}
            executed={play.display?.action ?? null}
          />
          <div className="tiles">
            <Tile
              label="Score"
              value={scoreFromDistance(play.state?.distance ?? 0).toLocaleString()}
            />
            <Tile label="Best" value={play.best.toLocaleString()} />
            <Tile label="Deaths" value={play.crashed ? '1' : '0'} />
            <Tile
              label="Answer time"
              value={play.p50 === null ? '—' : `${play.p50.toFixed(0)} ms`}
              sub={play.p95 === null ? undefined : `model ${play.p95.toFixed(0)} ms`}
            />
            <Tile
              label="Survival"
              value={`${(play.frames / 60).toFixed(1)}s`}
              sub={`${Math.round(play.state?.speed ?? 0)} px/s`}
            />
            <Tile
              label="Best move"
              value={`${play.rate.toFixed(0)}%`}
              sub={`${play.agreement.agree}/${play.agreement.checked} frames`}
            />
            <Tile label="Live saves" value="0" sub="no shield" />
            <Tile label="API cost" value="free" sub="on this host" />
          </div>
        </aside>
      </main>

      {play.diverged !== null ? (
        <div className="banner banner--danger" role="alert">
          The executed action diverged from the checkpoint argmax at frame {play.diverged}.
        </div>
      ) : null}
      {play.error ? (
        <div className="banner banner--danger" role="alert">
          {play.error}
        </div>
      ) : null}
      {play.crashed ? (
        <div className="banner banner--danger">
          Run ended at frame {play.frames.toLocaleString()}; no action was substituted.
        </div>
      ) : null}
      {play.completed ? (
        <div className="banner">
          Run reached the session limit at frame {play.frames.toLocaleString()} without a crash.
        </div>
      ) : null}

      <footer className="board__foot">
        <dl className="facts">
          <div>
            <dt>model</dt>
            <dd>{model.name ?? 'Tiny Dino Policy'}</dd>
          </div>
          <div>
            <dt>parameters</dt>
            <dd>{model.params.toLocaleString()}</dd>
          </div>
          <div>
            <dt>checkpoint</dt>
            <dd>{formatBytes(model.checkpoint_bytes)}</dd>
          </div>
          <div>
            <dt>engine</dt>
            <dd>{model.engine_revision}</dd>
          </div>
          <div>
            <dt>features</dt>
            <dd>{model.feature_version ?? '—'}</dd>
          </div>
          <div>
            <dt>device</dt>
            <dd>{model.device ?? '—'}</dd>
          </div>
          <div>
            <dt>interventions</dt>
            <dd>{model.runtime_interventions ?? 0}</dd>
          </div>
          <div>
            <dt>seed</dt>
            <dd>{play.seed}</dd>
          </div>
          <div className="facts__wide">
            <dt>sha-256</dt>
            <dd>{model.checkpoint_sha256}</dd>
          </div>
          <div className="facts__wide">
            <dt>runtime</dt>
            <dd>greedy argmax executed unchanged · no planner, no shield</dd>
          </div>
        </dl>
      </footer>
    </div>
  )
}

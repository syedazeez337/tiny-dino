/**
 * Geometry and drawing for the Dino arena.
 *
 * Every sprite rectangle below is a position in the Chromium offline-runner 2x sheet
 * (see `laya_mlx/trex/app.py` in the reference project). The painter draws at 2x and the
 * canvas is scaled by CSS, exactly like the reference window does.
 */

export const GAME_WIDTH = 600
export const GAME_HEIGHT = 150
export const GROUND_Y = 140
export const DINO_X = 60
export const PIXEL = 2

export const SPRITE_URL = '/dino/sprite-2x.png'

export const ACTION_ORDER = ['run', 'jump', 'duck'] as const
export type Action = (typeof ACTION_ORDER)[number]

/** Row order used by the reference arena's probability panel. */
export const DISPLAY_ACTIONS: readonly Action[] = ['jump', 'duck', 'run']

export const ACTION_KEYS: Record<Action, string> = {
  run: 'RUN',
  jump: 'JUMP ↑',
  duck: 'DUCK ↓',
}

/** The model's own choice, recomputed from its probabilities so the client can verify it. */
export function argmax(probabilities: number[]): Action | null {
  let best = Number.NEGATIVE_INFINITY
  let index = -1
  probabilities.forEach((value, position) => {
    if (value > best) {
      best = value
      index = position
    }
  })
  return index >= 0 ? (ACTION_ORDER[index] ?? null) : null
}

export interface Obstacle {
  kind: string
  x: number
  width: number
  height: number
  y: number
}

export interface EngineState {
  frame: number
  distance: number
  speed: number
  dino_y: number
  dino_vy: number
  ducking: boolean
  control: string
  motion: string
  obstacle: Obstacle | null
  next_obstacle: Obstacle | null
}

export interface Frame {
  action: string
  probabilities: number[]
  latency_ms: number
  state: EngineState
}

export interface Rect {
  sx: number
  sy: number
  sw: number
  sh: number
  dx: number
  dy: number
  dw: number
  dh: number
}

const TREX = { x: 1678, y: 2, run: [88, 132], jump: 0, duck: [264, 323] } as const

const OBSTACLE_SHEET: Record<string, { x: number; y: number; w: number; h: number }> = {
  small: { x: 446, y: 2, w: 34, h: 70 },
  large: { x: 652, y: 2, w: 50, h: 100 },
  bird: { x: 260, y: 2, w: 92, h: 80 },
}

const SCORE_FONT = { x: 1294, y: 2, w: 20, h: 26, advance: 22 }

/** The original distance meter: 0.025 score units per scrolled pixel. */
export function scoreFromDistance(distance: number): number {
  return distance > 0 ? Math.round(distance * 0.025) : 0
}

export function dinoRect(state: EngineState, tick: number): Rect {
  const ducking = state.ducking
  const airborne = state.dino_y > 0
  let index: number
  if (ducking) index = TREX.duck[Math.floor(tick / 8) % 2] ?? TREX.duck[0]
  else if (airborne) index = TREX.jump
  else index = TREX.run[Math.floor(tick / 6) % 2] ?? TREX.run[0]
  const sw = (ducking ? 59 : 44) * PIXEL
  const sh = 47 * PIXEL
  return {
    sx: TREX.x + index * PIXEL,
    sy: TREX.y,
    sw,
    sh,
    dx: DINO_X * PIXEL,
    dy: (GROUND_Y - state.dino_y - 47) * PIXEL,
    dw: sw,
    dh: sh,
  }
}

export function obstacleRect(obstacle: Obstacle, distance: number, tick: number): Rect {
  const sheet = OBSTACLE_SHEET[obstacle.kind] ?? OBSTACLE_SHEET.small
  const dx = (obstacle.x - distance) * PIXEL
  if (obstacle.kind === 'bird') {
    const frame = Math.floor(tick / 10) % 2
    return {
      sx: sheet.x + frame * sheet.w,
      sy: sheet.y,
      sw: sheet.w,
      sh: sheet.h,
      dx,
      dy: (GROUND_Y - obstacle.y - sheet.h / PIXEL) * PIXEL,
      dw: sheet.w,
      dh: sheet.h,
    }
  }
  return {
    sx: sheet.x,
    sy: sheet.y,
    sw: sheet.w,
    sh: sheet.h,
    dx,
    dy: (GROUND_Y - obstacle.height) * PIXEL,
    dw: sheet.w,
    dh: sheet.h,
  }
}

export function horizonRects(distance: number): Rect[] {
  const width = GAME_WIDTH * PIXEL
  const offset = -((distance * PIXEL) % width)
  return [0, width].map((base) => ({
    sx: 4,
    sy: 104,
    sw: width,
    sh: 24,
    dx: offset + base,
    dy: 127 * PIXEL,
    dw: width,
    dh: 24,
  }))
}

export function scoreRects(score: number, units = 5): Rect[] {
  const text = String(Math.max(0, score)).padStart(units, '0').slice(-units)
  const startX = GAME_WIDTH - 11 * (units + 1)
  return Array.from(text).map((char, index) => {
    const digit = Math.min(9, Math.max(0, char.charCodeAt(0) - 48))
    return {
      sx: SCORE_FONT.x + digit * SCORE_FONT.w,
      sy: SCORE_FONT.y,
      sw: SCORE_FONT.w,
      sh: SCORE_FONT.h,
      dx: (startX + 11 * index) * PIXEL,
      dy: 10 * PIXEL,
      dw: SCORE_FONT.w,
      dh: SCORE_FONT.h,
    }
  })
}

export function stableFrames(frame: Frame): Obstacle[] {
  const obstacles: Obstacle[] = []
  for (const candidate of [frame.state.obstacle, frame.state.next_obstacle]) {
    if (!candidate) continue
    const left = candidate.x - frame.state.distance
    if (left + candidate.width > 0 && left < GAME_WIDTH) obstacles.push(candidate)
  }
  return obstacles
}

/** One shared image instance: the live lane and the crash replay draw from the same sheet. */
let sheet: HTMLImageElement | null = null

export function loadSheet(url = SPRITE_URL): HTMLImageElement {
  if (!sheet) {
    sheet = new Image()
    sheet.src = url
  }
  return sheet
}

export function resetSheet(): void {
  sheet = null
}

export function drawGame(canvas: HTMLCanvasElement, frame: Frame | null, tick: number): void {
  let context: CanvasRenderingContext2D | null = null
  try {
    context = canvas.getContext('2d')
  } catch {
    return
  }
  if (!context) return
  const image = loadSheet()
  context.setTransform(1, 0, 0, 1, 0, 0)
  context.fillStyle = '#ffffff'
  context.fillRect(0, 0, canvas.width, canvas.height)
  if (!image.complete || image.naturalWidth === 0) return
  const blit = (rect: Rect) => {
    context?.drawImage(image, rect.sx, rect.sy, rect.sw, rect.sh, rect.dx, rect.dy, rect.dw, rect.dh)
  }
  if (!frame) return
  for (const rect of horizonRects(frame.state.distance)) blit(rect)
  for (const obstacle of stableFrames(frame)) blit(obstacleRect(obstacle, frame.state.distance, tick))
  blit(dinoRect(frame.state, tick))
  for (const rect of scoreRects(scoreFromDistance(frame.state.distance))) blit(rect)
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null
  const ordered = [...values].sort((a, b) => a - b)
  return ordered[Math.floor(ordered.length / 2)] ?? null
}

/** The next clean axis maximum at or above `value`, as the reference charts do. */
export function niceCeiling(value: number): number {
  if (value <= 0) return 1
  const power = 10 ** Math.floor(Math.log10(value))
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (step * power >= value) return step * power
  }
  return 10 * power
}

export function bucketMedians(
  samples: { t: number; v: number }[],
  bucketSeconds = 0.5,
): { t: number; v: number }[] {
  const buckets = new Map<number, number[]>()
  for (const sample of samples) {
    const key = Math.floor(sample.t / bucketSeconds)
    const bucket = buckets.get(key)
    if (bucket) bucket.push(sample.v)
    else buckets.set(key, [sample.v])
  }
  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([key, values]) => ({ t: key * bucketSeconds + bucketSeconds / 2, v: median(values) ?? 0 }))
}

import { describe, expect, it } from 'vitest'
import {
  bucketMedians,
  dinoRect,
  GROUND_Y,
  median,
  niceCeiling,
  obstacleRect,
  scoreFromDistance,
  scoreRects,
  stableFrames,
  type EngineState,
  type Frame,
} from '../app/arena'

const state = (over: Partial<EngineState> = {}): EngineState => ({
  frame: 0,
  distance: 0,
  speed: 360,
  dino_y: 0,
  dino_vy: 0,
  ducking: false,
  control: 'run',
  motion: 'grounded',
  obstacle: null,
  next_obstacle: null,
  ...over,
})

describe('score meter', () => {
  it('matches the original 0.025 units per pixel', () => {
    expect(scoreFromDistance(0)).toBe(0)
    expect(scoreFromDistance(4000)).toBe(100)
    expect(scoreFromDistance(1000)).toBe(25)
  })

  it('renders one glyph per score digit, right aligned', () => {
    const glyphs = scoreRects(7, 2)
    expect(glyphs).toHaveLength(2)
    expect(glyphs[0]?.sx).toBe(1294)
    expect(glyphs[1]?.sx).toBe(1294 + 7 * 20)
    expect(glyphs[1]?.dx).toBeGreaterThan(glyphs[0]?.dx ?? 0)
  })
})

describe('sprite geometry', () => {
  it('stands the dino on the ground line', () => {
    const rect = dinoRect(state(), 0)
    expect(rect.dx).toBe(60 * 2)
    expect(rect.dy).toBe((GROUND_Y - 47) * 2)
    expect(rect.sw).toBe(44 * 2)
  })

  it('lifts the dino by its height above ground', () => {
    const rect = dinoRect(state({ dino_y: 40 }), 0)
    expect(rect.dy).toBe((GROUND_Y - 40 - 47) * 2)
  })

  it('uses the wider duck sprite while ducking', () => {
    expect(dinoRect(state({ ducking: true }), 0).sw).toBe(59 * 2)
  })

  it('places obstacles relative to the scrolled distance', () => {
    const rect = obstacleRect({ kind: 'large', x: 600, width: 25, height: 50, y: 0 }, 300, 0)
    expect(rect.dx).toBe(300 * 2)
    expect(rect.dy).toBe((GROUND_Y - 50) * 2)
    expect(rect.sw).toBe(50)
  })

  it('anchors birds above the ground rather than on it', () => {
    const rect = obstacleRect({ kind: 'bird', x: 400, width: 46, height: 30, y: 75 }, 0, 0)
    expect(rect.dy).toBe((GROUND_Y - 75 - 40) * 2)
  })

  it('keeps only obstacles still on screen', () => {
    const frame: Frame = {
      action: 'run',
      probabilities: [1, 0, 0],
      latency_ms: 0,
      state: state({
        distance: 500,
        obstacle: { kind: 'small', x: 560, width: 17, height: 35, y: 0 },
        next_obstacle: { kind: 'small', x: 1300, width: 17, height: 35, y: 0 },
      }),
    }
    expect(stableFrames(frame)).toHaveLength(1)
  })
})

describe('chart helpers', () => {
  it('computes the median without reordering the input', () => {
    const values = [3, 1, 2]
    expect(median(values)).toBe(2)
    expect(values).toEqual([3, 1, 2])
    expect(median([])).toBeNull()
  })

  it('picks a clean axis ceiling', () => {
    expect(niceCeiling(1)).toBe(1)
    expect(niceCeiling(7)).toBe(10)
    expect(niceCeiling(230)).toBe(250)
  })

  it('buckets samples into median points', () => {
    const points = bucketMedians(
      [
        { t: 0.1, v: 10 },
        { t: 0.2, v: 20 },
        { t: 0.8, v: 40 },
      ],
      0.5,
    )
    expect(points).toHaveLength(2)
    expect(points[0]?.v).toBe(20)
    expect(points[1]?.v).toBe(40)
  })
})

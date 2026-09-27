import { describe, expect, it } from 'vitest'
import { tickLabel } from '../app/Charts'

const whole = (v: number) => v.toFixed(0)

describe('tickLabel', () => {
  it('keeps the chart format for integer ticks', () => {
    expect(tickLabel(200, whole)).toBe('200')
    expect(tickLabel(0, whole)).toBe('0')
  })

  it('does not round fractional ticks to a neighbouring integer', () => {
    expect(tickLabel(0.5, whole)).toBe('0.5')
    expect(tickLabel(1.25, whole)).toBe('1.25')
    expect(tickLabel(2.5, whole)).toBe('2.5')
  })
})

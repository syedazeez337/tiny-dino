import { niceCeiling } from './arena'

interface Point {
  t: number
  v: number
}

interface LineChartProps {
  title: string
  points: Point[]
  format: (value: number) => string
}

interface BarRow {
  label: string
  value: number | null
}

const WIDTH = 320
const HEIGHT = 112
const LEFT = 50
const RIGHT = WIDTH - 12
const TOP = 18
const BOTTOM = HEIGHT - 18

/** Integer ticks use the chart's format; fractional ticks such as 1.25 keep their decimals. */
export function tickLabel(value: number, format: (value: number) => string): string {
  return Number.isInteger(value) ? format(value) : String(Number(value.toFixed(2)))
}

/** A small line chart with three gridlines, matching the reference arena's charts. */
export function LineChart({ title, points, format }: LineChartProps) {
  const tMax = Math.max(20, points.length > 0 ? (points[points.length - 1]?.t ?? 0) : 0)
  const vMax = niceCeiling(Math.max(1, ...points.map((point) => point.v)))
  const x = (t: number) => LEFT + ((RIGHT - LEFT) * Math.min(t, tMax)) / tMax
  const y = (v: number) => BOTTOM - ((BOTTOM - TOP) * Math.min(v, vMax)) / vMax
  const path = points.map((point) => `${x(point.t).toFixed(1)},${y(point.v).toFixed(1)}`).join(' ')
  const last = points[points.length - 1]

  return (
    <figure className="chart">
      <figcaption>
        <span>{title}</span>
        <span className="chart__end">{last ? format(last.v) : '—'}</span>
      </figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={title}>
        {[0, 0.5, 1].map((fraction) => {
          const gridY = BOTTOM - (BOTTOM - TOP) * fraction
          return (
            <g key={fraction}>
              <line x1={LEFT} x2={RIGHT} y1={gridY} y2={gridY} className="chart__grid" />
              <text x={LEFT - 6} y={gridY + 3} className="chart__tick" textAnchor="end">
                {tickLabel(vMax * fraction, format)}
              </text>
            </g>
          )
        })}
        <text x={LEFT} y={HEIGHT - 4} className="chart__tick">
          0 s
        </text>
        <text x={RIGHT} y={HEIGHT - 4} className="chart__tick" textAnchor="end">
          {Math.round(tMax)} s
        </text>
        {points.length > 1 ? <polyline points={path} className="chart__line" /> : null}
        {last ? <circle cx={x(last.t)} cy={y(last.v)} r={3} className="chart__dot" /> : null}
      </svg>
    </figure>
  )
}

interface BarsProps {
  title: string
  rows: BarRow[]
  format: (value: number) => string
  max: number
}

/** Horizontal bars with a square base and a rounded data end, as the reference draws them. */
export function Bars({ title, rows, format, max }: BarsProps) {
  const viewHeight = 62
  const left = 12
  const span = WIDTH - left - 72
  const thickness = 14
  const rowTop = 18
  return (
    <figure className="chart">
      <figcaption>
        <span>{title}</span>
      </figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${viewHeight}`} role="img" aria-label={title}>
        {rows.map((row, index) => {
          const y = rowTop + index * 30
          const length = row.value === null ? 0 : Math.max(3, (span * Math.min(row.value, max)) / max)
          return (
            <g key={row.label}>
              <rect x={left} y={y} width={span} height={thickness} rx={4} className="chart__baseline" />
              {row.value !== null ? (
                <>
                  <rect x={left} y={y} width={length} height={thickness} rx={4} className="chart__bar" />
                  <text x={left + length + 8} y={y + 11} className="chart__tick">
                    {format(row.value)}
                  </text>
                </>
              ) : (
                <text x={left} y={y + 11} className="chart__tick">
                  —
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

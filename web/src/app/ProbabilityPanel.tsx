import { ACTION_KEYS, ACTION_ORDER, DISPLAY_ACTIONS } from './arena'

interface ProbabilityPanelProps {
  probabilities: number[]
  executed: string | null
}

/** The model's full distribution every frame; the executed action is highlighted. */
export function ProbabilityPanel({ probabilities, executed }: ProbabilityPanelProps) {
  return (
    <div className="probs" aria-label="Next move probabilities">
      <span className="probs__title">Next move · model probabilities</span>
      {DISPLAY_ACTIONS.map((action) => {
        const value = probabilities[ACTION_ORDER.indexOf(action)] ?? 0
        const chosen = executed === action
        return (
          <div className={chosen ? 'prob prob--chosen' : 'prob'} key={action}>
            <span className="prob__key">{ACTION_KEYS[action]}</span>
            <span className="prob__track">
              <span
                className="prob__fill"
                style={{ width: `${Math.min(100, Math.max(2, value * 100))}%` }}
              />
            </span>
            <span className="prob__value">{Math.round(value * 100)}%</span>
          </div>
        )
      })}
    </div>
  )
}

import { useState } from 'react'
import { DinoGame } from './DinoGame'
import { TicTacToe } from './TicTacToe'

type Game = 'dino' | 'ttt'

const GAMES: { id: Game; label: string }[] = [
  { id: 'dino', label: 'Dino' },
  { id: 'ttt', label: 'Tic-Tac-Toe' },
]

export function App() {
  const [game, setGame] = useState<Game>('dino')

  return (
    <div className="games">
      <header className="games__bar">
        <span className="games__chip" />
        <span className="games__brand">Tiny Games</span>
        <div className="segmented" role="tablist" aria-label="Choose a game">
          {GAMES.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={game === item.id}
              onClick={() => setGame(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <span className="games__tagline">two small policies · one interface</span>
      </header>
      <div className="games__body">{game === 'dino' ? <DinoGame /> : <TicTacToe />}</div>
    </div>
  )
}

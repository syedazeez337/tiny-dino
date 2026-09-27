import { useEffect, useRef, useState } from 'react'
import { drawGame, GAME_HEIGHT, GAME_WIDTH, loadSheet, PIXEL, type Frame } from './arena'

interface GameCanvasProps {
  frame: Frame | null
  tick?: number
  className?: string
}

/** The lane, drawn from the Chromium sprite sheet at 2x and scaled to fit by CSS. */
export function GameCanvas({ frame, tick = 0, className }: GameCanvasProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const image = loadSheet()
    if (image.complete && image.naturalWidth > 0) {
      setReady(true)
      return
    }
    const onLoad = () => setReady(true)
    image.addEventListener('load', onLoad)
    return () => image.removeEventListener('load', onLoad)
  }, [])

  useEffect(() => {
    if (canvas.current) drawGame(canvas.current, frame, tick)
  }, [frame, tick, ready])

  return (
    <canvas
      ref={canvas}
      className={className}
      width={GAME_WIDTH * PIXEL}
      height={GAME_HEIGHT * PIXEL}
      role="img"
      aria-label="Dino lane"
    />
  )
}

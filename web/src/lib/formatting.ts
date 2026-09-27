export function formatCount(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const kib = bytes / 1024
  if (kib < 1024) return `${kib.toFixed(2)} KiB`
  return `${(kib / 1024).toFixed(2)} MiB`
}

export function pluralize(count: number, one: string, many = `${one}s`): string {
  return count === 1 ? one : many
}

export function moveLabel(heap: number, newValue: number): string {
  return `Pile ${heap + 1}: ${newValue}`
}

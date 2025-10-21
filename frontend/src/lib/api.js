export const getApiBase = () =>
  (import.meta.env.VITE_API_BASE && String(import.meta.env.VITE_API_BASE).trim()) || 'http://localhost:8000'

export const buildAbsoluteUrl = (pathOrUrl) => {
  if (!pathOrUrl) return ''
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl
  const base = getApiBase().replace(/\/$/, '')
  const path = String(pathOrUrl).replace(/^\//, '')
  return `${base}/${path}`
}

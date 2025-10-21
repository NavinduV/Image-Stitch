import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ImagePreviewGrid from './components/ImagePreviewGrid.jsx'
import { buildAbsoluteUrl, getApiBase } from './lib/api.js'

export default function App() {
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [stitchedUrl, setStitchedUrl] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    const urls = files.map((f) => ({ url: URL.createObjectURL(f), name: f.name }))
    setPreviews(urls)
    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u.url))
    }
  }, [files])

  const onSelectFiles = useCallback((e) => {
    const list = Array.from(e.target.files || [])
    setError('')
    setStitchedUrl('')
    setFiles(list)
  }, [])

  const resetAll = useCallback(() => {
    setFiles([])
    setPreviews([])
    setStitchedUrl('')
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }, [])

  const canStitch = useMemo(() => files.length >= 2 && !loading, [files, loading])

  const onStitch = useCallback(async () => {
    if (files.length < 2) {
      setError('Please select at least 2 images.')
      return
    }
    setLoading(true)
    setError('')
    setStitchedUrl('')

    try {
      const form = new FormData()
      files.forEach((f) => form.append('files', f))

      const res = await fetch(`${getApiBase().replace(/\/$/, '')}/stitch/`, {
        method: 'POST',
        body: form,
      })

      const data = await res.json()
      if (!res.ok || data.error) {
        const msg = data?.error || 'Stitching failed'
        const details = data?.details ? `\nDetails: ${JSON.stringify(data.details)}` : ''
        throw new Error(`${msg}${details}`)
      }

      const absUrl = buildAbsoluteUrl(data.imageUrl)
      setStitchedUrl(absUrl)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }, [files])

  return (
    <div className="min-h-screen">
      <header className="bg-white border-b">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Image Stitcher</h1>
          <span className="text-xs text-gray-500">API: {getApiBase()}</span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-6 space-y-6">
        <section className="bg-white p-4 sm:p-6 rounded-xl shadow border space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Select images (2+)</label>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={onSelectFiles}
              className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
            />
            <p className="text-xs text-gray-500 mt-1">
              Images will be previewed locally. Nothing is uploaded until you click Stitch.
            </p>
          </div>

          <ImagePreviewGrid items={previews} />

          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={onStitch}
              disabled={!canStitch}
              className="inline-flex items-center px-4 py-2 rounded-lg bg-indigo-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Stitching…' : 'Stitch'}
            </button>
            <button
              onClick={resetAll}
              className="inline-flex items-center px-4 py-2 rounded-lg border bg-white hover:bg-gray-50"
            >
              Reset
            </button>
            {error && <span className="text-sm text-red-600 whitespace-pre-wrap">{error}</span>}
          </div>
        </section>

        {stitchedUrl && (
          <section className="bg-white p-4 sm:p-6 rounded-xl shadow border space-y-4">
            <h2 className="text-lg font-semibold">Result</h2>
            <div className="rounded-lg overflow-hidden border">
              <img src={stitchedUrl} alt="stitched" className="w-full max-h-[70vh] object-contain bg-black" />
            </div>
            <div className="flex items-center gap-3">
              <a
                href={stitchedUrl}
                download
                className="inline-flex items-center px-4 py-2 rounded-lg bg-emerald-600 text-white"
              >
                Download
              </a>
              <a
                href={stitchedUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center px-4 py-2 rounded-lg border bg-white hover:bg-gray-50"
              >
                Open in new tab
              </a>
            </div>
          </section>
        )}
      </main>

      <footer className="text-center text-xs text-gray-500 py-6">© {new Date().getFullYear()} Image Stitcher</footer>
    </div>
  )
}

import React from 'react'

export default function ImagePreviewGrid({ items = [] }) {
  if (!items.length) return null
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
      {items.map((it, idx) => (
        <div key={idx} className="rounded-lg overflow-hidden border bg-white shadow-sm">
          <img
            src={typeof it === 'string' ? it : it.url}
            alt={typeof it === 'string' ? `selected-${idx}` : it.name || `selected-${idx}`}
            className="w-full h-36 object-cover"
          />
          {typeof it !== 'string' && it.name && (
            <div className="px-2 py-1 text-xs text-gray-600 truncate">{it.name}</div>
          )}
        </div>
      ))}
    </div>
  )
}

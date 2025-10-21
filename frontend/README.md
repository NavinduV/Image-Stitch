# Image Stitcher Frontend (Vite + React + Tailwind)

A simple UI to upload multiple images, preview them, stitch via FastAPI backend, and download the result.

## Prerequisites
- Node.js 18+
- Backend running locally at `http://localhost:8000` (FastAPI from this repo)

## Configure API base
Create `.env.local` (already added):

```
VITE_API_BASE=http://localhost:8000
```

## Install & run (Windows PowerShell)

```
cd "c:\\Users\\Navindu\\Desktop\\Research_Dev\\image_stitch\\frontend"
npm install
npm run dev
```

Then open the URL shown (default http://localhost:5173).

## Build
```
npm run build
npm run preview
```

## Notes
- CORS must allow http://localhost:5173 in the backend (already configured in `stitch.py`).
- The backend serves stitched files under `/outputs`. The frontend turns the relative `imageUrl` into a full URL using `VITE_API_BASE`.

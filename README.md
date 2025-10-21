# Image Stitch - Deploy to Render

This project has two services:

- Backend: FastAPI (Python) in `backend/`
- Frontend: Vite + React in `frontend/`

## 1) Backend on Render (Python Web Service)

Files:
- `backend/requirements.txt`
- `backend/render.yaml` (service definition)

Steps:
1. Push to GitHub (repo public or private linked to Render).
2. In Render, click New + Blueprint and point to the repo. It will detect `backend/render.yaml`.
3. Confirm the service "image-stitch-backend" and deploy.

Notes:
- Start command: `uvicorn stitch:app --host 0.0.0.0 --port $PORT`
- Environment: `ALLOWED_ORIGINS` defaults to `*` (change to your frontend URL once known).

## 2) Frontend on Render (Static Site)

Files:
- `frontend/render.yaml` (service definition)

Steps:
1. Same blueprint will create the static service "image-stitch-frontend".
2. Build command: `npm ci && npm run build`
3. Publish path: `dist`
4. The `VITE_API_BASE` env var is dynamically set to the backend `RENDER_EXTERNAL_URL`.

## 3) Local development

Backend:
```powershell
.\env\Scripts\Activate.ps1
uvicorn backend.stitch:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:
```powershell
cd frontend
npm run dev
```

Set `frontend/.env.local`:
```
VITE_API_BASE=http://localhost:8000
```

## Troubleshooting
- If the frontend cannot reach the backend on Render, set `VITE_API_BASE` manually to the backend’s URL.
- Update CORS by setting `ALLOWED_ORIGINS` in the backend service to your frontend URL.
- OpenCV headless is used for servers: `opencv-python-headless`.

## Vercel deployment (alternative)

Backend (Serverless Function):
- Root Directory: `backend`
- Install Command: `pip install -r requirements.txt`
- Build/Output: leave empty
- Files added:
	- `vercel.json` routes all requests to `backend/api/index.py`
	- `backend/api/index.py` exports `app` from `stitch.py`
- Env:
	- `ALLOWED_ORIGINS=https://YOUR_FRONTEND_DOMAIN` (or `*` for testing)

Frontend (separate Vercel project):
- Root Directory: `frontend`
- Install Command: `npm ci`
- Build Command: `npm run build`
- Output Directory: `dist`
- Env:
	- `VITE_API_BASE=https://<your-backend>.vercel.app`

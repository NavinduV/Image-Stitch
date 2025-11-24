# stitch.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import cv2
import numpy as np
import os
import uuid

logger = logging.getLogger("stitch")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- Config (works without env vars; supports env override) ---
# Default allowed origin for local dev
DEFAULT_FRONTEND = "http://localhost:5173"   # change if your local frontend runs on another port
# Default backend url for building absolute image URLs
DEFAULT_BACKEND_URL = "http://localhost:8000"

# Read env overrides if present
origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
backend_url_env = os.environ.get("BACKEND_URL", "").strip()

if origins_env:
    # ALLOWED_ORIGINS can be comma-separated list
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    # default to the deployed frontend + local dev origin
    origins = [
        DEFAULT_FRONTEND,
        "https://image-stitch-theta.vercel.app",
    ]

BACKEND_URL = backend_url_env or os.environ.get("VERCEL_URL", "") or DEFAULT_BACKEND_URL
# if VERCEL_URL present it might be like "image-stitch-backend.vercel.app" so ensure scheme
if BACKEND_URL and BACKEND_URL.startswith("http") is False and "vercel.app" in BACKEND_URL:
    BACKEND_URL = "https://" + BACKEND_URL

# Configure CORS correctly: exact origins (good for credentialed requests)
# If someone explicitly sets ALLOWED_ORIGINS="*" then allow wildcard w/out credentials
if origins == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,   # set to True because we return exact origins
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Directory to serve stitched image outputs (ephemeral in serverless)
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# --- small constants ---
MAX_BYTES_PER_FILE = 12 * 1024 * 1024  # 12 MB per file (adjust as needed)

# --- helper functions (stitch logic preserved) ---
def _get_feature_extractor():
    # Prefer SIFT if available (OpenCV >= 4.4 has SIFT_create), else ORB
    if hasattr(cv2, "SIFT_create"):
        try:
            return cv2.SIFT_create()
        except Exception:
            pass
    return cv2.ORB_create(nfeatures=4000)

def _match_keypoints(desc1, desc2, use_sift: bool):
    if desc1 is None or desc2 is None:
        return []
    if use_sift:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def _estimate_homography(kp1, kp2, matches, ransac_thresh=4.0):
    if len(matches) < 4:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    return H

def _compute_cumulative_homographies(images):
    fx = _get_feature_extractor()
    use_sift = hasattr(cv2, "SIFT_create")
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    kps = []
    descs = []
    for g in grays:
        kp, des = fx.detectAndCompute(g, None)
        kps.append(kp)
        descs.append(des)

    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(images)):
        matches = _match_keypoints(descs[i], descs[i-1], use_sift)
        H_i_im1 = _estimate_homography(kps[i], kps[i-1], matches)
        if H_i_im1 is None:
            return None
        H_to_prev = Hs[i-1]
        H_to_0 = H_to_prev @ H_i_im1
        Hs.append(H_to_0)
    return Hs

def _warp_and_blend(images, Hs):
    corners = []
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(pts, H)
        corners.append(warped)
    all_pts = np.vstack(corners).reshape(-1,2)
    min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)
    shift_x = -min(0, min_x)
    shift_y = -min(0, min_y)
    out_w = max_x + shift_x
    out_h = max_y + shift_y
    if out_w <= 0 or out_h <= 0:
        return None
    T = np.array([[1,0,shift_x],[0,1,shift_y],[0,0,1]], dtype=np.float64)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    for im, H in zip(images, Hs):
        Ht = T @ H
        warped = cv2.warpPerspective(im, Ht, (out_w, out_h))
        mask = cv2.warpPerspective(np.ones(im.shape[:2], dtype=np.float32), Ht, (out_w, out_h))
        canvas += warped.astype(np.float32) * mask[..., None]
        weight += mask

    weight = np.clip(weight, 1e-6, None)
    out = (canvas / weight[..., None]).astype(np.uint8)
    return out

# --- endpoint ---
@app.post("/stitch/")
async def stitch_images(
    files: list[UploadFile] = File(...),
    mode: str = Form("panorama"),
    try_scans_on_fail: bool = Form(True),
    downscale: float = Form(1.0),
):
    images = []
    # read and validate
    for file in files:
        data = await file.read()
        if not data:
            continue
        if len(data) > MAX_BYTES_PER_FILE:
            return JSONResponse(status_code=400, content={"error": f"File {file.filename} too large (max {MAX_BYTES_PER_FILE} bytes)"})
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return JSONResponse(status_code=400, content={"error": "Need at least 2 images to stitch"})

    def resize_images(imgs, scale_val: float):
        if scale_val >= 0.999:
            return imgs
        out = []
        for im in imgs:
            h, w = im.shape[:2]
            new_w = max(1, int(w * scale_val))
            new_h = max(1, int(h * scale_val))
            if new_w == w and new_h == h:
                out.append(im)
            else:
                out.append(cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA))
        return out

    def run_stitch_for(mode_name: str, imgs):
        m = (mode_name or "").strip().lower()
        if m == "scans":
            st = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:
            st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        return st.stitch(imgs)

    status_map = {0: "OK", 1: "ERR_NEED_MORE_IMGS", 2: "ERR_HOMOGRAPHY_EST_FAIL", 3: "ERR_CAMERA_PARAMS_ADJUST_FAIL"}
    user_scale = downscale if isinstance(downscale, (int, float)) else 1.0
    try:
        user_scale = float(user_scale)
    except Exception:
        user_scale = 1.0
    scales = []
    if 0.1 <= user_scale <= 1.0:
        scales.append(round(user_scale, 2))
    for sc in [1.0, 0.7, 0.5, 0.35]:
        if sc not in scales:
            scales.append(sc)

    primary_mode = (mode or "panorama").strip().lower()
    modes = ["panorama", "scans"]
    if primary_mode in ("panorama", "scans"):
        other = "scans" if primary_mode == "panorama" else "panorama"
        modes = [primary_mode, other]

    orders = [False, True]
    attempts = []

    try:
        for sc in scales:
            imgs_scaled = resize_images(images, sc)
            for mo in modes:
                for rev in orders:
                    imgs_try = list(reversed(imgs_scaled)) if rev else imgs_scaled
                    status, out_img = run_stitch_for(mo, imgs_try)
                    attempts.append({"scale": sc, "mode": mo, "reversed": rev, "status": int(status), "statusText": status_map.get(int(status), str(status))})
                    if status == cv2.Stitcher_OK and out_img is not None:
                        filename = f"stitched_{uuid.uuid4().hex}.jpg"
                        output_path = os.path.join(OUTPUTS_DIR, filename)
                        cv2.imwrite(output_path, out_img)
                        image_url = f"{BACKEND_URL.rstrip('/')}/outputs/{filename}" if BACKEND_URL else f"/outputs/{filename}"
                        return {"message": "Stitching successful", "imageUrl": image_url, "usedMode": mo, "usedScale": sc, "reversedOrder": rev}
    except Exception as e:
        logger.exception("stitcher raised an exception")
        # fall through to fallback path

    # fallback custom homography
    try:
        sc_final = scales[-1]
        imgs_small = resize_images(images, sc_final)
        Hs = _compute_cumulative_homographies(imgs_small)
        if Hs is not None:
            blended = _warp_and_blend(imgs_small, Hs)
            if blended is not None:
                filename = f"stitched_{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(OUTPUTS_DIR, filename)
                cv2.imwrite(output_path, blended)
                image_url = f"{BACKEND_URL.rstrip('/')}/outputs/{filename}" if BACKEND_URL else f"/outputs/{filename}"
                return {"message": "Stitching successful (custom homography)", "imageUrl": image_url, "usedMode": "custom-homography", "usedScale": sc_final, "reversedOrder": False}
    except Exception:
        logger.exception("custom homography fallback failed")

    return JSONResponse(status_code=400, content={"error": "Stitching failed after multiple attempts. Ensure images overlap and try different capture order.", "details": {"attempts": attempts}})

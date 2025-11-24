import os
import sys

# Ensure parent directory (backend/) is on sys.path so we can import stitch.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from stitch import app  # FastAPI app

# Export the app for Vercel serverless functions
# Vercel looks for 'app' variable in the handler file


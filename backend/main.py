"""
IntelliReview — FastAPI Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from config import API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS
from routes.analyze import router as analyze_router
from routes.health import router as health_router

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# CORS — allow file:// and localhost origins for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permissive for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router, tags=["Health"])
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])

# Mount frontend static files
# Calculate absolute path to frontend directory (one level up from backend)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"⚠️ Warning: Frontend directory not found at {frontend_path}")


@app.on_event("startup")
async def startup_event():
    """Pre-warm the ML model on startup."""
    from ml.predict import load_model
    load_model()
    print("✅ IntelliReview API ready — model loaded.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

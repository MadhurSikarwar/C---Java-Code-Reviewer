"""
IntelliReview — FastAPI Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from config import API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS
from routes.analyze import router as analyze_router
from routes.health import router as health_router
from routes.models import router as models_router


# Lifespan event handler for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup."""
    # Startup: Load the ML model
    from ml.predict import load_model
    load_model()
    print("✅ IntelliReview API ready — model loaded.")

    # Load the neural networks (TensorFlow is slow to import) in the background so the first real request is not the one
    # that pays for it.
    import threading

    def _warm():
        try:
            from ml.predict import predict_risk
            predict_risk([0.0] * 36, model_type="all", issues=[])
            print("🔥 All four models warmed up.")
        except Exception as e:  # noqa: BLE001 - warm-up must never stop the server
            print(f"⚠️ Warm-up skipped: {e}")

    threading.Thread(target=_warm, daemon=True, name="model-warmup").start()
    
    yield  # Application runs here
    
    # Shutdown: Cleanup if needed
    print("🛑 IntelliReview API shutting down.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
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
app.include_router(models_router, prefix="/api", tags=["Models"])

# Mount frontend static files
# Calculate absolute path to frontend directory (one level up from backend)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"⚠️ Warning: Frontend directory not found at {frontend_path}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

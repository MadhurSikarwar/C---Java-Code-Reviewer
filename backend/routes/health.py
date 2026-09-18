"""
IntelliReview — /health route
"""
from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "IntelliReview API",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

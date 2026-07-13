from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from ...core.redis import redis
from ...core.device import get_device_status

router = APIRouter()

@router.get("/health")
async def health():
    try:
        pong = await redis.ping()
        return {
            "status": "ok",
            "redis": bool(pong),
            "model_runtime": get_device_status(),
        }
    except RedisError as e:
        # Return 503 if Redis isn’t reachable
        return JSONResponse(
            {
                "status": "degraded",
                "redis": False,
                "model_runtime": get_device_status(),
                "detail": str(e),
            },
            status_code=503,
        )

from fastapi import APIRouter

from routers import mi

router = APIRouter()

router.include_router(mi.router, prefix="/mi")

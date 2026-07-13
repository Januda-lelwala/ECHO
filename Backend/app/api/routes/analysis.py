import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.failure_discovery_service import DiscoveryRecord, discover_failures


router = APIRouter(prefix="/analysis")


class FailureDiscoveryRecord(BaseModel):
    id: str
    features: dict[str, Any]
    prediction: str
    ground_truth: str


class FailureDiscoveryRequest(BaseModel):
    task: Literal["classification", "transcription"]
    records: list[FailureDiscoveryRecord] = Field(min_length=20, max_length=10_000)
    min_slice_size: int = Field(default=5, ge=2, le=500)
    max_depth: int = Field(default=3, ge=1, le=5)


@router.post("/failure-discovery")
async def failure_discovery(request: FailureDiscoveryRequest):
    try:
        return await asyncio.to_thread(
            discover_failures,
            [
                DiscoveryRecord(
                    record_id=record.id,
                    features=record.features,
                    prediction=record.prediction,
                    ground_truth=record.ground_truth,
                )
                for record in request.records
            ],
            request.task,
            min_slice_size=request.min_slice_size,
            max_depth=request.max_depth,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

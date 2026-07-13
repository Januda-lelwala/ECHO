import pytest

from app.services.failure_discovery_service import DiscoveryRecord, discover_failures


def _classification_records(count: int = 80) -> list[DiscoveryRecord]:
    records = []
    for index in range(count):
        noisy_short = index % 4 == 0
        ground_truth = "happy" if index % 2 == 0 else "sad"
        prediction = "sad" if noisy_short and ground_truth == "happy" else ground_truth
        records.append(
            DiscoveryRecord(
                record_id=f"audio-{index}",
                features={
                    "duration": 0.8 if noisy_short else 3.2,
                    "environment": "noisy" if noisy_short else "clean",
                    "sample_rate": 16000,
                },
                prediction=prediction,
                ground_truth=ground_truth,
            )
        )
    return records


def test_discovers_repeatable_failure_slice():
    result = discover_failures(
        _classification_records(),
        "classification",
        min_slice_size=5,
    )

    assert result["record_count"] == 80
    assert result["baseline_error_rate"] > 0
    assert result["findings"]
    assert result["findings"][0]["error_lift"] > 1
    assert result["feature_importance"]


def test_requires_enough_labeled_records():
    with pytest.raises(ValueError, match="At least 20"):
        discover_failures(_classification_records(10), "classification")


@pytest.mark.asyncio
async def test_failure_discovery_endpoint(client):
    records = _classification_records()
    response = await client.post(
        "/analysis/failure-discovery",
        json={
            "task": "classification",
            "records": [
                {
                    "id": record.record_id,
                    "features": record.features,
                    "prediction": record.prediction,
                    "ground_truth": record.ground_truth,
                }
                for record in records
            ],
            "min_slice_size": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "classification"
    assert payload["findings"]

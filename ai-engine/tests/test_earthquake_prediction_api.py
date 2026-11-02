
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "latitude": 20.5,
    "longitude": 79.2,
    "year": 2026,
    "month": 11,
    "past_1m_count": 1,
    "past_3m_count": 2,
    "past_6m_count": 4,
}


def test_valid_prediction():
    response = client.post(
        "/api/v1/predictions/earthquake",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["grid_cell"]["grid_lat"] == 20
    assert data["grid_cell"]["grid_lon"] == 78


def test_invalid_latitude():
    payload = {**VALID_PAYLOAD, "latitude": 95}

    response = client.post(
        "/api/v1/predictions/earthquake",
        json=payload
    )

    assert response.status_code == 422


def test_invalid_longitude():
    payload = {**VALID_PAYLOAD, "longitude": 200}

    response = client.post(
        "/api/v1/predictions/earthquake",
        json=payload
    )

    assert response.status_code == 422


def test_invalid_month():
    payload = {**VALID_PAYLOAD, "month": 13}

    response = client.post(
        "/api/v1/predictions/earthquake",
        json=payload
    )

    assert response.status_code == 422


def test_negative_earthquake_count():
    payload = {**VALID_PAYLOAD, "past_1m_count": -1}

    response = client.post(
        "/api/v1/predictions/earthquake",
        json=payload
    )

    assert response.status_code == 422


def test_invalid_count_hierarchy():
    payload = {
        **VALID_PAYLOAD,
        "past_1m_count": 5,
        "past_3m_count": 2,
        "past_6m_count": 4
    }

    response = client.post(
        "/api/v1/predictions/earthquake",
        json=payload
    )

    assert response.status_code == 400

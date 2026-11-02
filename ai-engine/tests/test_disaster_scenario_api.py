from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "latitude": 20.5,
    "longitude": 79.2,
    "n_scenarios": 5,
    "scenario_name": "Nagpur Earthquake Simulation"
}


def test_generate_scenarios():
    response = client.post(
        "/api/v1/scenarios/generate",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    data = response.json()

    assert data["generated_scenarios"] == 5
    assert len(data["scenarios"]) == 5
    assert data["model"] == "V3 Increment Generator"


def test_scenario_grid_coordinates():
    response = client.post(
        "/api/v1/scenarios/generate",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    scenario = response.json()["scenarios"][0]

    assert scenario["grid_lat"] == 20
    assert scenario["grid_lon"] == 78
    assert scenario["latitude_range"] == [20, 22]
    assert scenario["longitude_range"] == [78, 80]


def test_count_hierarchy():
    response = client.post(
        "/api/v1/scenarios/generate",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    for scenario in response.json()["scenarios"]:
        assert (
            scenario["past_1m_count"]
            <= scenario["past_3m_count"]
            <= scenario["past_6m_count"]
        )


def test_generate_1000_scenarios():
    payload = {
        **VALID_PAYLOAD,
        "n_scenarios": 1000
    }

    response = client.post(
        "/api/v1/scenarios/generate",
        json=payload
    )

    assert response.status_code == 200
    assert response.json()["generated_scenarios"] == 1000


def test_reject_more_than_1000_scenarios():
    payload = {
        **VALID_PAYLOAD,
        "n_scenarios": 1001
    }

    response = client.post(
        "/api/v1/scenarios/generate",
        json=payload
    )

    assert response.status_code == 422


def test_reject_invalid_latitude():
    payload = {
        **VALID_PAYLOAD,
        "latitude": 95
    }

    response = client.post(
        "/api/v1/scenarios/generate",
        json=payload
    )

    assert response.status_code == 422

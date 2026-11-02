
import math
from app.core.count_generator_service import count_generator_service


class DisasterScenarioService:

    def generate(
        self,
        latitude: float,
        longitude: float,
        n_scenarios: int,
        scenario_name: str
    ):

        # Validate geographical coordinates
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")

        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")

        if not 1 <= n_scenarios <= 1000:
            raise ValueError("Generate between 1 and 1000 scenarios.")

        # Map coordinates to the existing 2-degree grid
        grid_resolution = 2

        grid_lat = float(
            math.floor(latitude / grid_resolution) * grid_resolution
        )

        grid_lon = float(
            math.floor(longitude / grid_resolution) * grid_resolution
        )

        # Generate synthetic historical count patterns using V3
        generated_records = count_generator_service.generate(n_scenarios)

        scenarios = []

        for index, record in enumerate(generated_records, start=1):

            scenarios.append({
                "scenario_id": index,
                "scenario_type": "synthetic_earthquake_count",
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "latitude_range": [
                    grid_lat,
                    grid_lat + grid_resolution
                ],
                "longitude_range": [
                    grid_lon,
                    grid_lon + grid_resolution
                ],
                "past_1m_count": record["past_1m_count"],
                "past_3m_count": record["past_3m_count"],
                "past_6m_count": record["past_6m_count"]
            })

        return {
            "model": "V3 Increment Generator",
            "scenario_name": scenario_name,
            "generation_method": (
                "Global empirical joint sampling of historical "
                "earthquake-count increments"
            ),
            "generated_scenarios": len(scenarios),
            "scenarios": scenarios,
            "disclaimer": (
                "These are synthetic historical count patterns, "
                "not actual earthquake events or future forecasts. "
                "The generated counts are not conditioned on the "
                "selected geographical location. Magnitudes, exact "
                "event dates and precise locations are not simulated."
            )
        }


disaster_scenario_service = DisasterScenarioService()

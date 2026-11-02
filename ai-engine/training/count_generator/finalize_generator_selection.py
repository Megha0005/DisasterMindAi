
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

comparison_path = (
    ROOT / "evaluation/count_generator/comparison/v3_v4_comparison.json"
)

if not comparison_path.exists():
    raise FileNotFoundError(
        f"Comparison report not found: {comparison_path}"
    )

with open(comparison_path, "r") as f:
    comparison = json.load(f)

output_dir = ROOT / "evaluation/count_generator/final"
output_dir.mkdir(parents=True, exist_ok=True)

selection = {
    "project": "DisasterMind AI",
    "evaluation_type": "Validation-based experimental generator selection",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "validation_period": "January 2020 - December 2022",
    "selected_primary_generator": "V4",
    "benchmark_generator": "V3",
    "selection_rationale": [
        "V4 has lower mean absolute correlation error.",
        "V4 uses geographic conditional joint empirical sampling.",
        "V3 provides a strong global empirical baseline.",
        "Both generators preserve count hierarchy.",
        "V3 performs slightly better on several marginal distribution metrics."
    ],
    "limitations": [
        "Both generators underestimate some validation means and tails.",
        "Neither generator has been evaluated on the untouched test period yet.",
        "V4 is an experimental geographic sampler, not a calibrated earthquake forecasting system.",
        "Historical sampling does not establish future earthquake probabilities."
    ],
    "validation_comparison": comparison,
    "test_evaluation_status": "NOT_STARTED",
    "final_selection_status": "PROVISIONAL"
}

output_path = output_dir / "generator_selection_report.json"

with open(output_path, "w") as f:
    json.dump(selection, f, indent=4)

print("Generator selection report saved.")
print("Primary experimental generator: V4")
print("Benchmark generator: V3")
print("Test evaluation: NOT STARTED")
print("Selection status: PROVISIONAL")
print("Report:", output_path)


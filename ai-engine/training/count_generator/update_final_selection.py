
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

report_path = (
    ROOT / "evaluation/count_generator/final/generator_selection_report.json"
)

test_path = (
    ROOT / "evaluation/count_generator/final/v3_v4_untouched_test_report.json"
)

with open(report_path, "r") as f:
    report = json.load(f)

with open(test_path, "r") as f:
    test = json.load(f)

report.update({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "selected_primary_generator": "V3",
    "benchmark_generator": "V4",
    "selection_status": "SELECTED_AFTER_COMPARATIVE_TEST_EVALUATION",
    "test_evaluation_status": "COMPLETED",
    "test_period": "January 2023 - September 2025",
    "test_evaluation": test,
    "final_decision_rationale": [
        "V3 achieved lower KS statistics for recent and older increments.",
        "V4 achieved a marginally lower KS statistic for previous increments.",
        "V3 achieved lower mean absolute correlation error.",
        "Both generators preserved count hierarchy.",
        "Both generators underestimated the frequency of large counts."
    ],
    "scientific_caveat": [
        "The test results were used for comparative model selection.",
        "They should not be presented as an independent unbiased estimate after model selection.",
        "No further tuning should be performed using this test dataset.",
        "An additional future-period holdout would be needed for independent confirmation."
    ]
})

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print("Final generator selection updated.")
print("Primary generator: V3")
print("Experimental alternative: V4")
print("Test evaluation: COMPLETED")
print("Further test-set tuning: PROHIBITED")
print("Report:", report_path)

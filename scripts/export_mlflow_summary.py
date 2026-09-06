"""Export the portable, non-secret experiment evidence from a local MLflow store."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from mlflow.tracking import MlflowClient


def export_summary(tracking_uri: str, experiment_name: str, output: Path) -> Path:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    records = []
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time ASC"]
    )
    for run in runs:
        histories = {
            key: [
                {"step": item.step, "value": item.value, "timestamp": item.timestamp}
                for item in client.get_metric_history(run.info.run_id, key)
            ]
            for key in sorted(run.data.metrics)
        }
        records.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "parameters": dict(sorted(run.data.params.items())),
                "metric_history": histories,
                "artifacts": sorted(item.path for item in client.list_artifacts(run.info.run_id)),
            }
        )
    payload = {
        "schema_version": "finland-geospatial-ai-mlflow-export/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "tracking_uri_kind": "local-sqlite",
        "experiment_name": experiment_name,
        "runs": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="finland-geospatial-ai-nls")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlflow-run-summary.json"))
    args = parser.parse_args()
    print(export_summary(args.tracking_uri, args.experiment, args.output))


if __name__ == "__main__":
    main()

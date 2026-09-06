.PHONY: install acquire data validate data-report baseline train-unet train-ablation train-modern select test lint typecheck serve docker

install:
	python -m pip install -e '.[dev]'

acquire:
	geoai-acquire --config configs/data/nls_l324.yaml

data:
	geoai-data --config configs/data/nls_l324.yaml

validate:
	geoai-validate data/manifests/nls-l324-2025-v1.json

data-report:
	geoai-dataset-report data/manifests/nls-l324-2025-v1.json

baseline:
	geoai-evaluate --manifest data/manifests/nls-l324-2025-v1.json --majority-baseline --split val

train-unet:
	geoai-train --config configs/models/unet.yaml --manifest data/manifests/nls-l324-2025-v1.json

train-ablation:
	geoai-train --config configs/models/unet_no_boundary_ignore.yaml --manifest data/manifests/nls-l324-2025-v1.json

train-modern:
	geoai-train --config configs/models/segformer_b0.yaml --manifest data/manifests/nls-l324-2025-v1.json

select:
	geoai-select artifacts/models/E1-unet-rgb.json artifacts/models/E2-unet-no-boundary-ignore.json artifacts/models/E3-segformer-b0-rgb.json

test:
	pytest --cov=finland_geospatial_ai --cov-report=term-missing

lint:
	ruff check src tests

typecheck:
	mypy src/finland_geoai src/finland_geospatial_ai

serve:
	uvicorn finland_geospatial_ai.api:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

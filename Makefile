.PHONY: data validate-data train-baseline train-modern evaluate report serve test lint

data:
	python -m finland_geoai.data.build --config configs/dataset.yaml

validate-data:
	python -m finland_geoai.data.validate --manifest artifacts/dataset-manifest-v1.json

train-baseline:
	python -m finland_geoai.training.run --config configs/unet.yaml

train-modern:
	python -m finland_geoai.training.run --config configs/deeplab.yaml

evaluate:
	python -m finland_geoai.evaluation.run --artifact artifacts/final-model.pt

report:
	python -m finland_geoai.evaluation.run --artifact artifacts/final-model.pt --figures

serve:
	uvicorn finland_geoai.inference.api:app --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check .

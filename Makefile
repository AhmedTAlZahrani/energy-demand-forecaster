.PHONY: setup forecast compare

setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

forecast:
	.venv/Scripts/python -m forecaster.main

compare:
	.venv/Scripts/python -m forecaster.main --compare

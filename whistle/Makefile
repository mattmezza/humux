.PHONY: test build release

test:
	uv venv --python 3.12 --allow-existing
	uv pip install -r requirements-dev.txt
	uv run --no-project ruff check .
	uv run --no-project pytest -q

build:
	docker build -t whistle:local .

release:
	uv run --no-project --python 3.12 python release.py "$(name)"

run:
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache **/__pycache__

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

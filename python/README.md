# qikits

A toolkit of curated algorithms, data structures, and reusable code snippets.

Install
-------

`pip install qikits`

Development
-----------

Using `uv` (recommended):

1. Install dependencies and the package in development mode:

	uv sync

2. Run tests:

	uv run pytest tests/ -q

3. Run an example notebook:

	uv run jupyter notebook notebooks/sampling/pyspark_weighted_sampling.ipynb

Manual setup (if not using `uv`):

1. Create & activate a virtual environment:

	python -m venv .venv
	source .venv/bin/activate

2. Install the package with test dependencies from `pyproject.toml`:

	pip install -e ".[test]"

3. Run tests:

	pytest tests/ -q

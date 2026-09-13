format:
	uv run ruff check . --fix

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$')

lint lint_diff:
	uv run ruff check .
	uv run ruff format .
	uv run mypy $(PYTHON_FILES)

test:
	uv run pytest -vv --cov=semantic_chunkers --cov-report=term-missing --cov-report=xml

test_functional:
	uv run pytest -vv -s --exitfirst --maxfail=1 tests/functional
test_unit:
	uv run pytest -vv --exitfirst --maxfail=1 tests/unit
test_integration:
	uv run pytest -vv --exitfirst --maxfail=1 tests/integration
# The report is rendered even when a variant failed, then the runner's exit
# code is preserved so CI still sees the failure. A failing render fails the
# recipe too, so a report that never got written cannot leave CI green; the
# runner's code wins when both fail, because it names the failing variant.
bench:
	uv run python -m benchmarks.run; status=$$?; \
	uv run python -m benchmarks.report; report=$$?; \
	if [ $$status -eq 0 ]; then status=$$report; fi; \
	exit $$status

bench_report:
	uv run python -m benchmarks.report --no-append

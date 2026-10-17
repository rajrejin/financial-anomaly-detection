.PHONY: gate
gate:
	ruff check . && ruff format --check . && mypy src && pytest

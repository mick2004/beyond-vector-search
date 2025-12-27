.PHONY: run eval test lint

run:
	PYTHONPATH=src python3 -m beyond_vector_search.run --query "What is adaptive retrieval?"

eval:
	PYTHONPATH=src python3 -m beyond_vector_search.evaluate

test:
	PYTHONPATH=src python3 -m unittest -q

lint:
	@echo "No linter enforced in this minimal repo (stdlib-only)."



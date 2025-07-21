.PHONY: install lint format test test-integration all

install:
	pipenv install --dev
	pre-commit install

lint:
	pipenv run pylint --recursive=y .

format:
	pipenv run black .
	pipenv run isort .

test:
	pipenv run pytest Tests/test_preprocessiong.py

test-integration:
	pipenv run pytest Tests/test_integration.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .pytest_cache

all: format lint test
